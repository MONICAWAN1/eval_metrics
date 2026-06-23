"""The input contract for evaluation: a real ``TargetSlide`` and the model's ``GeneratedNiches``.

The metrics take the real target slide and the generated cells — both as
plain arrays. 

Conventions
-----------
* Expression lives in the shared PCA space (``X_pca``) the generated cells were produced in.
* A niche = a centroid cell (point index 0) + its ``N - 1`` neighbours.
* Both sides must use the same PCA basis — guaranteed when ``TargetSlide`` is built from the
  same preprocessing ``.pkl`` whose ``X_pca`` the model was trained/generated on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetSlide:
    """A real target slide: every cell's expression + coordinates, plus optional labels/grid.

    Attributes:
        x: ``(N_cells, n_pcs)`` PCA expression of all real cells.
        pos: ``(N_cells, coord)`` spatial coordinates.
        ct: ``(N_cells,)`` integer cell-type labels (kept for reference/diagnostics; the
            concordance metric now uses a neutral classifier on paired niches, not these labels).
        grid_x / grid_pos: density-matched grid subsample used by Moran's I; if absent, the full
            cloud is used.
        n_classes: number of cell types (for proportion histograms).
    """

    x: np.ndarray
    pos: np.ndarray
    ct: np.ndarray | None = None
    grid_x: np.ndarray | None = None
    grid_pos: np.ndarray | None = None
    n_classes: int | None = None

    @property
    def moran_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """The (x, pos) used for the Moran's I real reference — the grid subsample if present."""
        if self.grid_x is not None and self.grid_pos is not None:
            return self.grid_x, self.grid_pos
        return self.x, self.pos

    @classmethod
    def from_dataclass(cls, ds, timepoint: str, n_pcs: int | None = None) -> "TargetSlide":
        """Build a ``TargetSlide`` from a preprocessing ``H5ADDatasetDataclass`` and a timepoint.

        Pass the target slide's own ``.pkl`` (e.g. ``target_abca.pkl``). Slices the given timepoint's 
        cells, maps cell-type labels to ints via ``ds.ct_to_int``, and extracts the precomputed 
        grid subsample (``subsampled_timepoint_idx``) as the Moran's I reference.
        """
        cells = np.asarray(ds.timepoint_indices[timepoint])
        x = np.asarray(ds.X_pca[cells])
        if n_pcs is not None:
            x = x[:, :n_pcs]
        pos = np.asarray(ds.coords[cells])

        ct_raw = np.asarray(ds.ct)[cells]
        if np.issubdtype(ct_raw.dtype, np.integer):
            ct = ct_raw.astype(np.int64)
        else:
            ct = np.array([ds.ct_to_int[c] for c in ct_raw], dtype=np.int64)
        n_classes = len(ds.ct_to_int)

        grid_x = grid_pos = None
        if ds.subsampled_timepoint_idx and timepoint in ds.subsampled_timepoint_idx:
            grid_idx = np.unique(np.asarray(ds.subsampled_timepoint_idx[timepoint]))
            grid_x, grid_pos = x[grid_idx], pos[grid_idx]

        return cls(x=x, pos=pos, ct=ct, grid_x=grid_x, grid_pos=grid_pos, n_classes=n_classes)


@dataclass
class GeneratedNiches:
    """Generated microenvironments: ``(B, N, D)`` with the centroid at point index 0.

    Attributes:
        x: ``(B, N, n_pcs)`` generated expression.
        pos: ``(B, N, coord)`` generated coordinates.
        gt_x / gt_pos: the matched ground-truth (paired real) target microenvironment per
            generated niche; supply these to enable the pointwise regression metrics
            (``x/*``, ``pos/*``) and the cell-type concordance metric (``ct/*``), which scores the
            generated niche against this paired real niche with a neutral classifier.
    """

    x: np.ndarray
    pos: np.ndarray
    gt_x: np.ndarray | None = None
    gt_pos: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x)
        self.pos = np.asarray(self.pos)
        if self.x.ndim != 3 or self.pos.ndim != 3:
            raise ValueError(
                f"GeneratedNiches expects (B, N, D) arrays; got x{self.x.shape} pos{self.pos.shape}"
            )

    @property
    def flat_x(self) -> np.ndarray:
        """All generated cells pooled into one ``(B*N, n_pcs)`` cloud."""
        return self.x.reshape(-1, self.x.shape[-1])

    @property
    def flat_pos(self) -> np.ndarray:
        """All generated cells pooled into one ``(B*N, coord)`` cloud."""
        return self.pos.reshape(-1, self.pos.shape[-1])

    @property
    def centroid_x(self) -> np.ndarray:
        """Centroid expression of each niche, ``(B, n_pcs)`` (point 0)."""
        return self.x[:, 0, :]

    @property
    def centroid_pos(self) -> np.ndarray:
        """Centroid coordinates of each niche, ``(B, coord)`` (point 0)."""
        return self.pos[:, 0, :]

    @classmethod
    def from_trajectory(cls, x_traj, pos_traj, **kwargs) -> "GeneratedNiches":
        """Build from a flow sampling trajectory: take the final step ``[-1]`` of each."""
        import numpy as _np

        def _last(t):
            arr = t[-1]
            return arr.detach().cpu().numpy() if hasattr(arr, "detach") else _np.asarray(arr)

        return cls(x=_last(x_traj), pos=_last(pos_traj), **kwargs)
