"""OT-CFM (fm_mnist) generation adapter.

A thin :class:`BaseGenerator` over the existing fm_mnist OT-CFM — it reuses fm_mnist's own code
(``VelocityMLP`` / ``sample_prior`` / ``midpoint_sample`` / ``invert_pca_expression``) and only adds
glue. It loads the checkpoint, samples the prior through the learned velocity field, inverts the
whitened-PCA back to gene space, and returns a comparable ``(target, generated)`` pair.

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
        n_pcs: components for the shared PCA fit on the reference (target).
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

        import pandas as pd
        import torch

        if self.fm_root:
            sys.path.insert(0, str(self.fm_root))
        try:
            from fm.data import invert_coords, invert_pca_expression, sample_prior
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

        # Naive spatial baseline: a checkpoint trained with --spatial_key emits
        # [expression | coords]; split off the coord tail and un-standardize it to real coords.
        coord_dim = int(stats.get("coord_dim", 0))
        gen_pos = None
        if self.coord_mode == "generate":
            if not coord_dim:
                raise ValueError(
                    "coord_mode='generate' but the checkpoint has no coord_dim — it was trained "
                    "expression-only. Retrain with `--spatial_key spatial`, or use coord_mode "
                    "'reference'/'random'."
                )
            gen, gen_pos = gen[:, :-coord_dim], invert_coords(gen[:, -coord_dim:], stats)
        gen_counts = invert_pca_expression(gen, stats)  # (n_gen, n_model_genes)

        # Align the model's gene panel to the reference's var order so the shared PCA projects right.
        gen_var = pd.Index([str(v) for v in stats["var_names"]])
        idx = gen_var.get_indexer([str(v) for v in tgt.var_names])
        if (idx < 0).any():
            missing = [str(v) for v, j in zip(tgt.var_names, idx) if j < 0][:5]
            raise ValueError(
                f"Reference has genes not in the OT-CFM panel (e.g. {missing}); use the slide the "
                "OT-CFM was trained on as the reference."
            )
        gen_counts = gen_counts[:, idx]

        if self.coord_mode == "generate":
            # the model produced its own coordinates (un-standardized above)
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

        # Return GENE-space cells (no projection): the evaluator reconciles feature space (fit a
        # raw-gene PCA via `evaluate --n_pcs <N>`, or project through a shared-PCA target pickle via
        # `evaluate --shared_pca`). Pre-projecting here would NOT match a separately-fit target PCA
        # (different rotation/sign) and breaks the standalone evaluate step.
        return from_generated_arrays(gen_counts, pos, tgt, ct_key=self.ct_key, n_pcs=None)
