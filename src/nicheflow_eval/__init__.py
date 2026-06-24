"""nicheflow-eval — standalone evaluation metrics for spatial single-cell generation.

Inputs are **original AnnData (``.h5ad``) files** — raw gene expression + spatial coordinates.

Standalone (evaluate cells you already generated)::

    from nicheflow_eval import TargetSlide, GeneratedNiches, evaluate

    target = TargetSlide.from_anndata("target.h5ad", ct_key="class")   # raw genes + obsm['spatial']
    generated = GeneratedNiches.from_anndata("generated.h5ad")         # (B, N, D), centroid first
    results = evaluate(target, generated)                              # {test/group/metric: value}

Full pipeline (checkpoint + raw slides -> generate -> metrics). The generation step is a
pluggable blackbox: pass any ``generator`` implementing :class:`nicheflow_eval.pipeline.Generator`.
The bundled NicheFlow adapter (needs the ``[pipeline]`` extra) is one such generator::

    from nicheflow_eval.pipeline import run_pipeline
    from nicheflow_eval.adapters.nicheflow import nicheflow_generator

    res = run_pipeline(
        "source.h5ad", "target.h5ad", "flow.ckpt",
        generator=nicheflow_generator, classifier_h5ad="clf.h5ad",
    )

Bring your own model: write a ``generator`` that returns a
:class:`nicheflow_eval.pipeline.GenerationOutput` (``from_generated_anndata`` does this in one line
from a generated ``.h5ad``) — no NicheFlow needed.
"""

from nicheflow_eval.contract import GeneratedNiches, TargetSlide
from nicheflow_eval.evaluate import ALL_GROUPS, evaluate

__all__ = ["ALL_GROUPS", "GeneratedNiches", "TargetSlide", "evaluate"]
