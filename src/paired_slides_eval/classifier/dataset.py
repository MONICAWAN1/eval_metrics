from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from paired_slides_eval.data.dataclass import load_h5ad_dataset_dataclass, slide_expression_matrix
from paired_slides_eval.metrics._common import knn_indices
from paired_slides_eval.utils.log import RankedLogger

_logger = RankedLogger(__name__, rank_zero_only=True)


class CellTypeBatch(TypedDict):
    X: torch.Tensor
    y: torch.Tensor


class H5ADCTDataset(Dataset):
    def __init__(self, filepath: str) -> None:
        ds = load_h5ad_dataset_dataclass(filepath=filepath)

        # Whitened X_pca — the shared basis the models are scored in, so the probe matches eval.
        self.X = torch.Tensor(slide_expression_matrix(ds))

        # The cell types
        ct_to_int_vec = np.vectorize(ds.ct_to_int.get)
        self.ct = torch.Tensor(ct_to_int_vec(ds.ct)).to(torch.long)

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, index: int) -> CellTypeBatch:
        if index > len(self) or index < 0:
            raise IndexError(f"Index {index} out of bounds [0, {len(self)})")

        return {"X": self.X[index], "y": self.ct[index]}


class SpatialH5ADCTDataset(Dataset):
    """Cell-type classification over *microenvironments* — expression-only KNN.

    Each item is a centroid cell together with its ``k`` nearest spatial neighbours (KNN by
    coordinates), represented as an **expression-only** point set ``(k+1, n_pcs)`` with the centroid
    at point 0. Coordinates are used only to pick the KNN membership and are then **discarded**, so
    the classifier is coordinate-blind (it sees a set of neighbour gene-expression vectors). The
    label is the centroid cell's type, predicted from its neighbours (the net masks point 0).

    The local spatial organisation is evaluated *implicitly* through which cells are in the set: a
    realistic generated cell should have a neighbourhood-expression signature like its nearest real
    cell's. Neighbour gathering is lazy in ``__getitem__`` so each cell's expression is stored once.

    ``target`` selects the supervision: ``"ct"`` (default) gives the centroid's cell-type label;
    ``"expr"`` gives the centroid's expression vector, turning this into a **masked-centroid
    expression regression** dataset (predict the masked centroid's expression from its neighbours).

    """

    def __init__(self, filepath: str, n_neighbors: int = 10, target: str = "ct") -> None:
        if target not in ("ct", "expr"):
            raise ValueError(f"target must be 'ct' or 'expr', got {target!r}")
        self.target = target
        ds = load_h5ad_dataset_dataclass(filepath=filepath)
        ct_to_int_vec = np.vectorize(ds.ct_to_int.get)

        # k = number of neighbours (niche size k+1). KNN always returns a full k (unlike the radius
        # graph), so the only clamp is to the cells available on a slide.
        self.n_neighbors = int(n_neighbors)

        # Keep per-slide expression + labels; gather neighbourhoods lazily by precomputed KNN indices.
        self.x_by_t: dict[str, torch.Tensor] = {}
        self.ct_by_t: dict[str, torch.Tensor] = {}
        self.neighbor_idx_by_t: dict[str, torch.Tensor] = {}
        # Flat index -> (timepoint, centroid local id)
        self.index: list[tuple[str, int]] = []

        expr = slide_expression_matrix(ds)  # shared whitened X_pca
        for timepoint in ds.timepoints_ordered:
            indices = ds.timepoint_indices[timepoint]
            self.x_by_t[timepoint] = torch.as_tensor(expr[indices], dtype=torch.float32)
            self.ct_by_t[timepoint] = torch.as_tensor(
                ct_to_int_vec(ds.ct[indices]),
                dtype=torch.long,
            )
            # (n_centroids, k+1) KNN local indices on this slide's coords (self at column 0).
            neighbor_idx = knn_indices(np.asarray(ds.coords)[indices], self.n_neighbors)
            self.neighbor_idx_by_t[timepoint] = torch.as_tensor(neighbor_idx, dtype=torch.long)
            self.index.extend((timepoint, i) for i in range(neighbor_idx.shape[0]))

        self.point_dim = self.x_by_t[ds.timepoints_ordered[0]].size(-1)  # expression only

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> CellTypeBatch:
        if index >= len(self) or index < 0:
            raise IndexError(f"Index {index} out of bounds [0, {len(self)})")

        timepoint, centroid = self.index[index]
        nbr = self.neighbor_idx_by_t[timepoint][centroid]  # (k+1,) centroid at 0
        points = self.x_by_t[timepoint][nbr]  # (k+1, n_pcs) expression only

        # "ct" -> centroid cell-type label; "expr" -> centroid expression vector (the masked target).
        y = (
            self.x_by_t[timepoint][centroid]
            if self.target == "expr"
            else self.ct_by_t[timepoint][centroid]
        )
        return {"X": points, "y": y}
