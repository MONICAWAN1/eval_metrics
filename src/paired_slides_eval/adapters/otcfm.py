"""OT-CFM (fm_mnist) generation adapter.

A thin :class:`BaseGenerator` over the existing fm_mnist OT-CFM — it reuses fm_mnist's own code
(``VelocityMLP`` / ``sample_prior`` / ``midpoint_sample``) and only adds glue. It loads the
checkpoint, samples the prior through the learned velocity field, and returns a comparable
``(target, generated)`` pair with the expression in the model's whitened-PCA space (the space it
trained in).

The OT-CFM is trained **in the shared whitened-PCA(50) basis** (via fm_mnist's
``train_cfm_spatial.py --pca_stats``, which injects NicheFlow's shared PCA), so its generated cells
already live in the target's basis: the evaluator passes them straight through — no inversion, no
reprojection (see :func:`paired_slides_eval.contract._pca_aware_transform`).

The OT-CFM is **unconditional**. By default it is **expression-only** (it generates whitened-PCA
expression, no coordinates): ``source`` is ignored and generated cells get **placeholder**
coordinates from the reference slide, so only the expression metrics (``distribution``, ``c2st``)
are meaningful — the spatial metrics just confirm the absence of structure.

A checkpoint trained with fm_mnist's ``--spatial_key`` (the **naive spatial baseline**) instead
emits ``[expression | coords]`` jointly; selecting ``coord_mode="generate"`` reads off that
coordinate tail and un-standardizes it to real coordinates, so the spatial metrics
(``moran``/``psd``/``spd``/``ot_*/pos``) become a genuine read on the baseline's geometry.

Selected with ``generator=otcfm`` (``configs/generator/otcfm.yaml``). Needs fm_mnist's ``fm`` package
importable — either ``uv pip install -e ../fm_mnist`` or set ``fm_root`` to the fm_mnist repo path.
"""

from __future__ import annotations

import numpy as np

from paired_slides_eval.adapters.base import BaseGenerator
from paired_slides_eval.pipeline.run import GenerationOutput, from_generated_arrays


class OTCFMGenerator(BaseGenerator):
    """Generate cells from an fm_mnist OT-CFM checkpoint and pair them with the reference slide.

    Construction parameters (from ``configs/generator/otcfm.yaml``):
        sample_n: number of cells to generate (0 -> match the reference cell count).
        sample_steps: midpoint-ODE integration steps.
        n_pcs: unused — the adapter emits the model's own PCA space and the shared-PCA(50) basis is
            fit at evaluate time (kept for config back-compat).
        ct_key: ``obs`` column with cell types on the reference (for the ``ct/*`` groups).
        coord_mode: how generated cells get coordinates — ``"generate"`` reads the **model's own**
            coordinate tail (only valid for a checkpoint trained with ``--spatial_key``: the naive
            spatial baseline); ``"reference"`` (resample the reference positions) or ``"random"``
            (uniform over the reference bounding box) are placeholders for an aspatial OT-CFM.
        spatial_key: ``obsm`` key for the reference's coordinates.
        device / seed: torch device and RNG seed.
        fm_root: path to the fm_mnist repo, prepended to ``sys.path`` so ``import fm`` works when
            fm_mnist is not pip-installed. ``None`` assumes ``fm`` is already importable.
    """

    def __init__(
        self,
        *,
        sample_n: int = 0,
        sample_steps: int = 100,
        n_pcs: int = 50,
        ct_key: str | None = "class",
        coord_mode: str = "reference",
        spatial_key: str = "spatial",
        device: str = "cpu",
        seed: int = 0,
        fm_root: str | None = None,
    ) -> None:
        self.sample_n = sample_n
        self.sample_steps = sample_steps
        self.n_pcs = n_pcs
        self.ct_key = ct_key
        self.coord_mode = coord_mode
        self.spatial_key = spatial_key
        self.device = device
        self.seed = seed
        self.fm_root = fm_root

    def __call__(self, *, source, target, checkpoint, **_) -> GenerationOutput:
        import sys

        import torch

        if self.fm_root:
            sys.path.insert(0, str(self.fm_root))
        try:
            from fm.data import invert_coords, sample_prior
            from fm.networks import VelocityMLP
            from fm.sampling import midpoint_sample
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "fm_mnist's `fm` package is not importable. `uv pip install -e ../fm_mnist`, or set "
                "`fm_root` in configs/generator/otcfm.yaml to the fm_mnist repo path."
            ) from exc

        from paired_slides_eval.data.anndata import read_anndata

        # Rebuild the trained model from the checkpoint (fm_mnist's own format).
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg, stats = ckpt["config"], ckpt["stats"]
        model = VelocityMLP(dim=cfg["dim"], hidden=cfg["hidden"]).to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        tgt = read_anndata(target)
        n_gen = self.sample_n if self.sample_n > 0 else tgt.n_obs

        # Sample: prior -> velocity-field ODE -> whitened PCA (+ coord tail when spatial).
        torch.manual_seed(self.seed)
        prior = sample_prior(n_gen, shape=(cfg["dim"],), device=self.device)
        with torch.no_grad():
            gen = midpoint_sample(model, prior.clone(), steps=self.sample_steps)
        gen = gen.cpu().numpy()

        # Naive spatial baseline: a checkpoint trained with --spatial_key emits [expression | coords].
        # Split off the coord tail. If it was trained in the shared coord frame (coord_frame='shared',
        # standardised with the target's stats), emit those standardised coords as-is — symmetric with
        # the shared-PCA expression, so the eval passes them through. A legacy (self-frame) checkpoint
        # is un-standardised back to raw coords for the eval to reconcile.
        coord_dim = int(stats.get("coord_dim", 0))
        gen_pos = None
        if self.coord_mode == "generate":
            if not coord_dim:
                raise ValueError(
                    "coord_mode='generate' but the checkpoint has no coord_dim — it was trained "
                    "expression-only. Retrain with `--spatial_key spatial`, or use coord_mode "
                    "'reference'/'random'."
                )
            gen, coord_tail = gen[:, :-coord_dim], gen[:, -coord_dim:]
            if stats.get("coord_frame") == "shared":
                gen_pos = coord_tail
            else:
                gen_pos = invert_coords(coord_tail, stats)

        if self.coord_mode == "generate":
            # the model produced its own coordinates (shared-frame standardised, or raw for legacy)
            pos = gen_pos
        else:
            # the OT-CFM is aspatial -> placeholder coordinates from the reference.
            rng = np.random.default_rng(self.seed)
            ref_pos = np.asarray(tgt.obsm[self.spatial_key], dtype=float)
            if self.coord_mode == "reference":
                pos = ref_pos[rng.integers(0, len(ref_pos), n_gen)]
            else:  # "random" over the reference bounding box
                lo, hi = ref_pos.min(axis=0), ref_pos.max(axis=0)
                pos = rng.uniform(lo, hi, size=(n_gen, ref_pos.shape[1]))

        # The model is trained in the shared whitened-PCA(50) basis (via fm_mnist's --pca_stats), so
        # `gen` is already in the target's space: the evaluator passes it straight through.
        return from_generated_arrays(gen, pos, tgt, ct_key=self.ct_key, n_pcs=None)
