"""``H5ADPreprocessor`` — port of ``nicheflow.preprocessing.h5ad_preprocessor``.

Turns an ``AnnData`` whose ``obsm['X_pca']`` / ``varm['PCs']`` are already computed (see
:func:`nicheflow_eval.adapters.nicheflow.preprocess.compute_pca`) into the
:class:`~nicheflow_eval.data.dataclass.H5ADDatasetDataclass` the niche-based code consumes:
per-slide standardized coordinates + standardized ``X_pca``, the radius graph, and the
density-matched grid subsample of centroids.

Faithful to the NicheFlow original, **minus the global-alignment paths** (PASTE2): coordinates are
standardized per slide and no cross-slide coupling is precomputed (generation falls back to its
original minibatch-OT pairing). The ``external_*`` arguments are kept — they let a held-out
classifier-training slide reuse the source+target label vocabulary and ``X_pca`` scaling.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from nicheflow_eval.adapters.nicheflow.graph import (
    chunked_cdist_sum_argsort,
    grid_based_sampling_by_y,
)
from nicheflow_eval.data.dataclass import H5ADDatasetDataclass

MIN_COVERAGE = 5
_logger = logging.getLogger(__name__)


class H5ADPreprocessor:
    def __init__(
        self,
        timepoint_column: str,
        cell_type_column: str,
        timepoints_ordered: list[Any],
        standardize_coordinates: bool = True,
        radius: float = 0.15,
        dx: float = 0.15,
        dy: float = 0.2,
        device: str = "cpu",
        chunk_size: int = 1000,
        # For preparing a held-out classifier-training slide in the SAME feature space as
        # an already-built (source+target) dataset: reuse that dataset's cell-type vocabulary
        # and X_pca standardisation instead of recomputing them. ``obsm["X_pca"]`` must already
        # be projected into the shared PCA basis by the caller.
        external_ct_ordered: list[Any] | None = None,
        external_x_pca_stats: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.timepoint_column = timepoint_column
        self.ct_column = cell_type_column
        self.timepoints_ordered = timepoints_ordered
        self.standardize_coordinates = standardize_coordinates
        self.external_ct_ordered = external_ct_ordered
        self.external_x_pca_stats = external_x_pca_stats
        self.radius = radius
        self.dx = dx
        self.dy = dy
        self.chunk_size = chunk_size
        self.device = torch.device(device)

        # Set during preprocessing
        self.timepoint_to_int = None
        self.timepoint_indices: dict[str, np.ndarray] = {}
        self.ct_ordered = None
        self.ct_to_int = None
        self.ct = None
        self.X_pca = None
        self.coords = None
        self.PCs = None

        self.subsampled_timepoint_idx: dict[Any, list[int]] = {}
        self.timepoint_neighboring_indices: dict[Any, dict[int, list[int]]] = {}
        self.timepoint_num_neighbors: dict[Any, int] = {}
        self.test_microenvs = None
        self.stats = {"coords": {}, "X_pca": {}}

    def preprocess_data(self, adata) -> None:
        self.PCs = adata.varm["PCs"]
        self._prepare_timepoints_and_annotations(adata)
        self._normalize_coordinates_and_features(adata)
        self._compute_radius_graphs()
        self._subsample_centroids()

    def _prepare_timepoints_and_annotations(self, adata) -> None:
        self.timepoint_to_int = {
            timepoint: i for i, timepoint in enumerate(self.timepoints_ordered)
        }
        self.timepoint_indices = {
            t: np.where(adata.obs[self.timepoint_column] == t)[0] for t in self.timepoints_ordered
        }
        # Reuse an external vocabulary when projecting a slide into an already-built dataset's
        # label space (so the classifier's classes line up).
        if self.external_ct_ordered is not None:
            self.ct_ordered = list(self.external_ct_ordered)
        else:
            self.ct_ordered = sorted(adata.obs[self.ct_column].cat.categories)
        self.ct_to_int = {annotation: i for i, annotation in enumerate(self.ct_ordered)}

    def _normalize_coordinates_and_features(self, adata) -> None:
        method = "standardization" if self.standardize_coordinates else "min-max scaling"
        _logger.info(f"Preprocessing the spatial coordinates per timepoint with: {method}")

        self.coords = np.asarray(adata.obsm["spatial"], dtype=np.float64).copy()

        for timepoint in self.timepoints_ordered:
            indices = self.timepoint_indices[timepoint]
            if self.standardize_coordinates:
                coords_mean = self.coords[indices].mean(axis=0)
                coords_std = self.coords[indices].std(axis=0)
                self.stats["coords"][timepoint] = {"mean": coords_mean, "std": coords_std}
                self.coords[indices] = (self.coords[indices] - coords_mean) / coords_std
            else:
                coords_min = self.coords[indices].min(axis=0)
                coords_max = self.coords[indices].max(axis=0)
                self.stats["coords"][timepoint] = {"min": coords_min, "max": coords_max}
                self.coords[indices] = (self.coords[indices] - coords_min) / (
                    coords_max - coords_min
                )

        self._normalize_features(adata)

    def _normalize_features(self, adata) -> None:
        # Reuse an external dataset's X_pca mean/std when projecting a slide into its feature
        # space; otherwise compute it from this data.
        if self.external_x_pca_stats is not None:
            X_pca_mean = self.external_x_pca_stats["mean"]  # noqa: N806
            X_pca_std = self.external_x_pca_stats["std"]  # noqa: N806
        else:
            X_pca_mean = adata.obsm["X_pca"].mean(axis=0)  # noqa: N806
            X_pca_std = adata.obsm["X_pca"].std(axis=0)  # noqa: N806

        self.stats["X_pca"] = {"mean": X_pca_mean, "std": X_pca_std}
        self.X_pca = (np.asarray(adata.obsm["X_pca"]) - X_pca_mean) / X_pca_std
        self.ct = np.array(adata.obs[self.ct_column])

    def _compute_radius_graphs(self) -> None:
        if self.device.type == "cpu":
            _logger.warning("Using `CPU`! Might be too slow!")

        compute_iter = tqdm(self.timepoints_ordered)
        for timepoint in compute_iter:
            compute_iter.set_description(f"Computing radius graphs for timepoint: '{timepoint}'")

            indices = self.timepoint_indices[timepoint]
            coords_t = torch.Tensor(self.coords[indices]).to(self.device)

            # Fix the number of nodes per graph.
            num_neighbors, C_t_argsorted = chunked_cdist_sum_argsort(  # noqa: N806
                coords=coords_t, radius=self.radius, chunk_size=self.chunk_size
            )
            unique, counts = torch.unique(num_neighbors, return_counts=True)

            # Choose the most common N for the current radius.
            N = unique[counts.argmax()].cpu().numpy()  # noqa: N806
            self.timepoint_num_neighbors[timepoint] = N

            neighbor_indices = C_t_argsorted[:, :N].cpu().numpy()
            neighbors_dict = {i: row for i, row in enumerate(neighbor_indices)}
            self.timepoint_neighboring_indices[timepoint] = neighbors_dict

            del coords_t, num_neighbors, C_t_argsorted, unique, counts
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _subsample_centroids(self) -> None:
        if self.coords is None:
            raise ValueError("The coordinates must not be None at this point.")

        subsample_iter = tqdm(self.timepoints_ordered)
        for timepoint in subsample_iter:
            subsample_iter.set_description(
                f"Subsampling centroids t='{timepoint}' | dx={self.dx} | dy={self.dy}"
            )
            gt_indices = self.timepoint_indices[timepoint]
            gt = self.coords[gt_indices]

            pos_idx = grid_based_sampling_by_y(coords=gt, dx=self.dx, dy=self.dy)
            self.subsampled_timepoint_idx[timepoint] = pos_idx

            # Validation
            subgraph_indices = np.unique(
                np.concatenate(
                    [self.timepoint_neighboring_indices[timepoint][idx] for idx in pos_idx]
                )
            )
            diff = np.abs(len(gt) - len(subgraph_indices))
            if diff > MIN_COVERAGE:
                _logger.warning(
                    "You should change the values for `dx` and `dy`."
                    + f"GT: {len(gt)} | Microenvironment cover: {len(subgraph_indices)}"
                )

        # Fix nodes by upsampling
        self.test_microenvs = max([len(x) for x in self.subsampled_timepoint_idx.values()])
        _logger.info(f"Fixing test microenvironments to {self.test_microenvs} per slice.")

        for timepoint in self.timepoints_ordered:
            length = len(self.timepoint_neighboring_indices[timepoint])
            subsampled_indices = self.subsampled_timepoint_idx[timepoint]
            n_upsample = max(self.test_microenvs - len(subsampled_indices), 0)

            if n_upsample != 0:
                choices = [i for i in range(length) if i not in subsampled_indices]
                choices = np.random.choice(choices, n_upsample, replace=False)
                self.subsampled_timepoint_idx[timepoint] = np.concatenate(
                    [self.subsampled_timepoint_idx[timepoint], choices]
                )

    def to_dataclass(self) -> H5ADDatasetDataclass:
        """Pack the preprocessed arrays into the standalone ``H5ADDatasetDataclass``."""
        return H5ADDatasetDataclass(
            X_pca=self.X_pca,
            coords=self.coords,
            ct=self.ct,
            PCs=self.PCs,
            timepoints_ordered=self.timepoints_ordered,
            timepoint_column=self.timepoint_column,
            timepoint_to_int=self.timepoint_to_int,
            timepoint_indices=self.timepoint_indices,
            ct_column=self.ct_column,
            ct_ordered=self.ct_ordered,
            ct_to_int=self.ct_to_int,
            timepoint_neighboring_indices=self.timepoint_neighboring_indices,
            timepoint_num_neighbors=self.timepoint_num_neighbors,
            subsampled_timepoint_idx=self.subsampled_timepoint_idx,
            standardize_coordinates=self.standardize_coordinates,
            radius=self.radius,
            dx=self.dx,
            dy=self.dy,
            stats=self.stats,
            test_microenvs=self.test_microenvs,
            aligned=False,
            pair_target_to_source=None,
        )

    def save(self, filepath: str) -> None:
        fp = Path(filepath)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open("wb") as file:
            pickle.dump(self.to_dataclass(), file, protocol=pickle.HIGHEST_PROTOCOL)
        _logger.info(f"Saved file at: {fp}")
