"""Generate cells from a trained NicheFlow checkpoint — ``nicheflow`` used as a blackbox.

This is the only place that imports the flow model. Given a preprocessed source+target dataclass
(from :func:`paired_slides_eval.adapters.nicheflow.preprocess.preprocess_pair`) and a checkpoint,
it builds the ``PointCloudFlow``, loads the trained backbone weights, and runs the original
``flow.sample`` over NicheFlow's ``TestMicroEnvDataset`` to produce the generated target niches.

The generated cells live in the preprocessor's **standardized ``X_pca``** space (the space the
model trained in), so they are directly comparable to the ``TargetSlide`` the adapter builds from
the **same** dataclass (see ``paired_slides_eval.adapters.nicheflow.target_from_dataclass``).

Requires the ``nicheflow`` package (the ``[pipeline]`` extra).
"""

from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paired_slides_eval.contract import GeneratedNiches


@dataclass
class GenerationResult:
    """The generated niches plus the paired real ground truth, all as numpy ``(B, N, D)`` arrays."""

    x: np.ndarray  # (B, N, pca_dim) generated expression (standardized X_pca space)
    pos: np.ndarray  # (B, N, coord_dim) generated coordinates
    gt_x: np.ndarray  # (B, N, pca_dim) paired real target microenvironments
    gt_pos: np.ndarray  # (B, N, coord_dim)
    gt_ct: np.ndarray  # (B,) true cell-type label of each paired real centroid

    def to_generated_niches(self) -> GeneratedNiches:
        return GeneratedNiches(
            x=self.x, pos=self.pos, gt_x=self.gt_x, gt_pos=self.gt_pos, gt_ct=self.gt_ct
        )

    def to_anndata(self):
        """Flatten the generated cells into an ``AnnData`` (one row per cell, niche id in obs).

        Layout matches :meth:`GeneratedNiches.from_anndata`: ``obs['niche_id']`` groups each
        niche's points (centroid first), ``obsm['spatial']`` the coords, and the paired real
        niches in ``obsm['gt_x']`` / ``obsm['gt_pos']`` with ``obs['gt_ct']`` the centroid label.
        """
        import anndata as ad
        import pandas as pd

        b, n, _ = self.x.shape
        niche_id = np.repeat(np.arange(b), n)
        adata = ad.AnnData(X=self.x.reshape(b * n, -1).astype(np.float32))
        adata.obs["niche_id"] = niche_id
        adata.obs["gt_ct"] = np.repeat(self.gt_ct, n)
        adata.obsm["spatial"] = self.pos.reshape(b * n, -1).astype(np.float32)
        adata.obsm["gt_x"] = self.gt_x.reshape(b * n, -1).astype(np.float32)
        adata.obsm["gt_pos"] = self.gt_pos.reshape(b * n, -1).astype(np.float32)
        adata.obs = adata.obs.astype({"niche_id": "int64", "gt_ct": "int64"})
        adata.obs.index = pd.RangeIndex(b * n).astype(str)
        return adata


def _to_nicheflow_dataclass(ds):
    """Re-wrap our standalone dataclass into ``nicheflow``'s so its dataset consumes it cleanly."""
    from dataclasses import fields

    from nicheflow.preprocessing import H5ADDatasetDataclass as NFDataclass

    return NFDataclass(**{f.name: getattr(ds, f.name) for f in fields(ds)})


def _build_flow(pca_dim, coord_dim, ohe_dim, *, num_steps, solver, variant):
    """Construct a ``PointCloudFlow`` matching the trained backbone (defaults match the configs)."""
    from nicheflow.models.backbones.pc_transformer import PointCloudTransformer
    from nicheflow.models.flows import CFM, VFM, PointCloudFlow

    backbone = PointCloudTransformer(
        pca_dim=pca_dim, ohe_dim=ohe_dim, coord_dim=coord_dim, output_dim=pca_dim + coord_dim
    )
    variants = {"cfm": CFM, "vfm": VFM}
    if variant not in variants:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {list(variants)}.")
    var = variants[variant](lambda_features=1.0, lambda_pos=1.0)
    return PointCloudFlow(backbone=backbone, variant=var, num_steps=num_steps, solver=solver)


def _load_backbone(flow, checkpoint: str, device: str) -> None:
    """Load trained weights into ``flow.backbone`` — handles bare state_dict or Lightning ckpt."""
    import torch

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        flow.backbone.load_state_dict(state)
        return
    except (RuntimeError, KeyError):
        pass
    # Lightning checkpoint: keys are prefixed (e.g. ``flow.backbone.<...>``).
    prefix = "flow.backbone."
    stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not stripped:
        prefix = "backbone."
        stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    flow.backbone.load_state_dict(stripped)


def generate(
    ds,
    checkpoint: str,
    *,
    target_timepoint: str | None = None,
    n_slices: int | None = None,
    num_steps: int = 20,
    solver: str = "euler",
    variant: str = "cfm",
    device: str = "cpu",
) -> GenerationResult:
    """Run the flow on the target slide of ``ds`` and return the generated niches + paired GT.

    Args:
        ds: a preprocessed source+target ``H5ADDatasetDataclass`` (from ``preprocess_pair``).
        checkpoint: path to the trained flow checkpoint (bare backbone state_dict or Lightning).
        target_timepoint: which slide to generate (default: the last in ``timepoints_ordered``).
        n_slices: one-hot dimension the backbone was trained with (default: number of slides in
            ``ds``). Must match the checkpoint's ``ohe_dim``.
        num_steps / solver / variant: sampler settings (defaults match the NicheFlow MBA config).
        device: torch device for the model + sampling.
    """
    import torch
    from nicheflow.datasets.microenv_dataset import TestMicroEnvDataset
    from nicheflow.utils.dataloading import microenv_val_collate
    from torch.utils.data import DataLoader

    t2 = target_timepoint or ds.timepoints_ordered[-1]
    pca_dim = int(ds.X_pca.shape[1])
    coord_dim = int(ds.coords.shape[1])
    ohe_dim = n_slices or len(ds.timepoints_ordered)

    # Write the (nicheflow-typed) dataclass to a temp pkl for TestMicroEnvDataset to load.
    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = Path(tmp) / "pair.pkl"
        with pkl_path.open("wb") as fh:
            pickle.dump(_to_nicheflow_dataclass(ds), fh, protocol=pickle.HIGHEST_PROTOCOL)

        flow = _build_flow(
            pca_dim, coord_dim, ohe_dim, num_steps=num_steps, solver=solver, variant=variant
        )
        _load_backbone(flow, checkpoint, device)
        flow.to(device).eval()

        test_ds = TestMicroEnvDataset(data_fp=str(pkl_path), upsample_factor=1)
        loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, collate_fn=microenv_val_collate
        )

        xs, poss, gt_xs, gt_poss = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                x_traj, pos_traj = flow.sample(batch=batch)
                xs.append(x_traj[-1].cpu().numpy())
                poss.append(pos_traj[-1].cpu().numpy())
                gt_xs.append(batch["X_t2"].cpu().numpy())
                gt_poss.append(batch["pos_t2"].cpu().numpy())

    x = np.concatenate(xs, axis=0)
    pos = np.concatenate(poss, axis=0)
    gt_x = np.concatenate(gt_xs, axis=0)
    gt_pos = np.concatenate(gt_poss, axis=0)

    # True label of each paired real centroid: the target slide's cell types at the grid centroids
    # (TestMicroEnvDataset orders niches by subsampled_timepoint_idx[t2]).
    tp_idx = np.asarray(ds.timepoint_indices[t2])
    ct_raw = np.asarray(ds.ct)[tp_idx]
    if np.issubdtype(ct_raw.dtype, np.integer):
        ct_int = ct_raw.astype(np.int64)
    else:
        ct_int = np.array([ds.ct_to_int[c] for c in ct_raw], dtype=np.int64)
    centroid_ids = np.asarray(ds.subsampled_timepoint_idx[t2], dtype=np.int64)
    gt_ct = ct_int[centroid_ids][: x.shape[0]]

    return GenerationResult(x=x, pos=pos, gt_x=gt_x, gt_pos=gt_pos, gt_ct=gt_ct)
