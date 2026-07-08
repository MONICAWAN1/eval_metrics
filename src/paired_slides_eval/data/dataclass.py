"""The preprocessed-slide dataclass used by the NicheFlow pipeline.

This is a verbatim port of ``nicheflow.preprocessing.h5ad_dataset_type``. The metrics repo
*reuses* the ``.pkl`` produced by NicheFlow preprocessing as the target slide — it carries
``X_pca`` (the shared PCA space generated cells also live in), ``coords``, ``ct``, the
per-timepoint cell indices, the precomputed neighbour graph, and the grid subsample. Only the
schema + loader are needed here; the scanpy-based builder (``h5ad_preprocessor``) is *not*
ported, because evaluation reuses an existing ``.pkl`` rather than re-deriving it.

"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class H5ADDatasetDataclass:
    # === Data === #
    X_pca: np.ndarray
    coords: np.ndarray
    ct: np.ndarray
    PCs: np.ndarray

    # === Timepoint info === #
    timepoints_ordered: list[str]
    timepoint_column: str
    timepoint_to_int: dict[str, int]
    timepoint_indices: dict[str, np.ndarray]

    # === Cell type info === #
    ct_column: str
    ct_ordered: list[str]
    ct_to_int: dict[str, int]

    # === Radius-Graph-Related === #
    timepoint_neighboring_indices: dict[str, dict[int, list[int]]]
    timepoint_num_neighbors: dict[str, int]
    subsampled_timepoint_idx: dict[str, list[int]]

    # === Parameters === #
    standardize_coordinates: bool
    radius: float
    dx: float
    dy: float

    # === Statistics == #
    stats: dict[str, dict[str, dict[str, np.ndarray]] | dict[str, np.ndarray]]

    # === Test-related == #
    test_microenvs: int

    # === Global-alignment-related (optional) === #
    aligned: bool = False
    pair_target_to_source: dict[tuple[str, str], np.ndarray] | None = None

    # === Shared-PCA recipe reconstruction (optional) === #
    # Enough to rebuild the gene -> X_pca transform (``SharedGenePCA``) from this pickle alone, so
    # cells generated in *gene space* (e.g. the OT-CFM baseline) can be projected into the very same
    # whitened PCA basis the niche model trained in. ``None`` on pickles predating this (older pkls
    # simply cannot back a ``SharedGenePCA`` and must be re-preprocessed to gain one).
    #   lognorm_mean: per-gene mean of the log-normalised fit data (the PCA centering vector).
    #   lognorm_target_sum: the ``normalize_total`` target used at fit time (median of fit per-cell
    #       totals); reused so new cells are normalised to the *same* scale, not their own median.
    #   var_names: the gene-panel order the PCA was fit on; new cells are reordered to it.
    lognorm_mean: np.ndarray | None = None
    lognorm_target_sum: float | None = None
    var_names: list | None = None


def slide_expression_matrix(ds) -> np.ndarray:
    """The per-cell expression matrix to score in: the shared whitened ``X_pca``.

    Both models generate directly in this basis (NicheFlow natively; OT-CFM via the injected
    shared PCA), so target, probes, and generated cells all live in one space.
    """
    return np.asarray(ds.X_pca)


def load_h5ad_dataset_dataclass(filepath: str) -> H5ADDatasetDataclass:
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"The file does not exist: {fp}")

    with fp.open("rb") as file:
        ds: H5ADDatasetDataclass = pickle.load(file)

    return ds
