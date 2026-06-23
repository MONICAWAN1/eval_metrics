"""nicheflow-eval — standalone evaluation metrics for spatial single-cell generation.

Inputs are **original AnnData (``.h5ad``) files** — raw gene expression + spatial coordinates.

Standalone (evaluate cells you already generated)::

    from nicheflow_eval import TargetSlide, GeneratedNiches, evaluate

    target = TargetSlide.from_anndata("target.h5ad", ct_key="class")   # raw genes + obsm['spatial']
    generated = GeneratedNiches.from_anndata("generated.h5ad")         # (B, N, D), centroid first
    results = evaluate(target, generated)                              # {test/group/metric: value}

Full pipeline (checkpoint + raw slides -> generate -> metrics; needs the ``[pipeline]`` extra)::

    from nicheflow_eval.pipeline import run_pipeline

    res = run_pipeline("source.h5ad", "target.h5ad", "flow.ckpt", classifier_h5ad="clf.h5ad")
"""

from nicheflow_eval.contract import GeneratedNiches, TargetSlide
from nicheflow_eval.evaluate import ALL_GROUPS, evaluate

__all__ = ["ALL_GROUPS", "GeneratedNiches", "TargetSlide", "evaluate"]
