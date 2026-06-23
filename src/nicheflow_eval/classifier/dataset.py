from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from nicheflow_eval.data.dataclass import load_h5ad_dataset_dataclass
from nicheflow_eval.utils.log import RankedLogger

_logger = RankedLogger(__name__, rank_zero_only=True)


class CellTypeBatch(TypedDict):
    X: torch.Tensor
    y: torch.Tensor


class H5ADCTDataset(Dataset):
    def __init__(self, filepath: str) -> None:
        ds = load_h5ad_dataset_dataclass(filepath=filepath)

        # PCA-reduced gene expressions
        self.X = torch.Tensor(ds.X_pca)

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
    """Cell-type classification over *microenvironments*.

    Each item is a centroid cell together with its ``n_neighbors`` nearest spatial
    neighbors (from the precomputed radius graph), represented as a point set of
    ``[gene_expression | relative_position]`` rows where the relative position is the
    neighbor's coordinate minus the centroid's. The centroid itself is the first
    point (relative position 0). The label is the centroid cell's type.

    Unlike :class:`H5ADCTDataset` (gene expression only), this feeds both expression
    and local spatial structure to the classifier. It is consumed by the same
    :class:`~nicheflow.tasks.ct_classification.CellTypeClassification` task: the batch
    key is still ``"X"`` (now shape ``(n_neighbors, n_pcs + coord_dim)``), so the task
    needs no change.

    Neighbour gathering is done lazily in ``__getitem__`` so we keep one copy of each
    cell's expression rather than ``n_neighbors`` copies in overlapping microenvs.
    """

    def __init__(self, filepath: str, n_neighbors: int = 32) -> None:
        ds = load_h5ad_dataset_dataclass(filepath=filepath)
        ct_to_int_vec = np.vectorize(ds.ct_to_int.get)

        # Neighbor count is fixed per slide; clamp K to the smallest so every item
        # has the same number of points and the default collate can stack them.
        min_n = min(int(v) for v in ds.timepoint_num_neighbors.values())
        self.n_neighbors = min(n_neighbors, min_n)
        if self.n_neighbors < n_neighbors:
            _logger.warning(
                f"Requested n_neighbors={n_neighbors} but the smallest slide only has "
                f"{min_n} neighbours per microenvironment. Using {self.n_neighbors}."
            )

        # Keep per-slide tensors; gather neighbourhoods lazily
        self.x_by_t: dict[str, torch.Tensor] = {}
        self.pos_by_t: dict[str, torch.Tensor] = {}
        self.ct_by_t: dict[str, torch.Tensor] = {}
        self.neighbor_idx_by_t: dict[str, torch.Tensor] = {}
        # Flat index -> (timepoint, centroid local id)
        self.index: list[tuple[str, int]] = []

        for timepoint in ds.timepoints_ordered:
            indices = ds.timepoint_indices[timepoint]
            self.x_by_t[timepoint] = torch.as_tensor(ds.X_pca[indices], dtype=torch.float32)
            self.pos_by_t[timepoint] = torch.as_tensor(ds.coords[indices], dtype=torch.float32)
            self.ct_by_t[timepoint] = torch.as_tensor(
                ct_to_int_vec(ds.ct[indices]), dtype=torch.long
            )

            neighbors = ds.timepoint_neighboring_indices[timepoint]
            # (n_centroids, N) of distance-sorted local indices (self at column 0);
            # keep the K nearest
            neighbor_idx = torch.as_tensor(
                np.stack([neighbors[i] for i in range(len(neighbors))]), dtype=torch.long
            )[:, : self.n_neighbors]
            self.neighbor_idx_by_t[timepoint] = neighbor_idx

            self.index.extend((timepoint, i) for i in range(neighbor_idx.size(0)))

        self.point_dim = self.x_by_t[ds.timepoints_ordered[0]].size(-1) + self.pos_by_t[
            ds.timepoints_ordered[0]
        ].size(-1)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> CellTypeBatch:
        if index >= len(self) or index < 0:
            raise IndexError(f"Index {index} out of bounds [0, {len(self)})")

        timepoint, centroid = self.index[index]
        nbr = self.neighbor_idx_by_t[timepoint][centroid]  # (K,)

        nbr_x = self.x_by_t[timepoint][nbr]  # (K, n_pcs)
        rel_pos = self.pos_by_t[timepoint][nbr] - self.pos_by_t[timepoint][centroid]  # (K, coord)
        points = torch.cat([nbr_x, rel_pos], dim=-1)  # (K, n_pcs + coord_dim)

        return {"X": points, "y": self.ct_by_t[timepoint][centroid]}
