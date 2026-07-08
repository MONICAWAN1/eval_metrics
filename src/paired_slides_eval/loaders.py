"""Load generated cells from a file, auto-detecting niche-shaped vs flat.

Reads ``.h5ad`` / ``.npz`` / ``.pkl`` into a :class:`~paired_slides_eval.contract.GeneratedNiches`
or :class:`~paired_slides_eval.contract.GeneratedSlide` — the inverse of
:func:`paired_slides_eval.pipeline.io.write_generated`.

"""

from __future__ import annotations

import numpy as np

from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide
from paired_slides_eval.data.anndata import read_anndata


def _generated_from_mapping(m) -> GeneratedNiches | GeneratedSlide:
    """Build generated cells from an ``x``/``pos`` mapping (``.npz`` or an
    unpickled dict).

    Niche-shaped if ``x`` is 3-D ``(B, N, D)`` (optionally with ``gt_x``/``gt_pos``/``gt_ct``),
    else a flat ``GeneratedSlide`` from 2-D ``x``/``pos``.

    """
    x = np.asarray(m["x"])
    if x.ndim == 3:
        extra = {k: np.asarray(m[k]) for k in ("gt_x", "gt_pos", "gt_ct") if k in m}
        return GeneratedNiches(x=x, pos=np.asarray(m["pos"]), **extra)
    return GeneratedSlide(x=x, pos=np.asarray(m["pos"]))


def _load_generated(path: str, *, niche_key: str = "niche_id") -> GeneratedNiches | GeneratedSlide:
    """Load generated cells, auto-detecting niche-shaped vs flat.

    ``.h5ad``: niche-shaped if ``obs[niche_key]`` is present, else a flat ``GeneratedSlide``.
    ``.npz``: niche-shaped if ``x`` is 3-D ``(B, N, D)`` (optionally with ``gt_x``/``gt_pos``/
    ``gt_ct``), else a flat ``GeneratedSlide`` from 2-D ``x``/``pos``.
    ``.pkl``: a generator result object (any object with a ``to_generated_niches`` method) or a
    dict with the same ``x``/``pos``[/``gt_*``] arrays as the ``.npz`` form.

    """
    if str(path).endswith(".h5ad"):
        adata = read_anndata(path)
        if niche_key in adata.obs:
            return GeneratedNiches.from_anndata(adata, niche_key=niche_key)
        return GeneratedSlide.from_anndata(adata)

    if str(path).endswith(".pkl"):
        import pickle

        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if hasattr(obj, "to_generated_niches"):  # a generator result object
            return obj.to_generated_niches()
        if isinstance(obj, (GeneratedNiches, GeneratedSlide)):
            return obj
        if isinstance(obj, dict) and "x" in obj and "pos" in obj:
            return _generated_from_mapping(obj)
        raise ValueError(
            f"Unrecognised generated .pkl contents ({type(obj).__name__}). Expected a "
            "GenerationResult, a GeneratedNiches/GeneratedSlide, or a dict with x/pos arrays. "
            "A preprocessed-slide pickle is a *real* slide — load it as a target with "
            "TargetSlide.from_dataclass instead.",
        )

    return _generated_from_mapping(np.load(path))
