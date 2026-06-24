"""paired-slides-eval — standalone evaluation metrics for spatial single-cell generation.

Inputs are **original AnnData (``.h5ad``) files** — raw gene expression + spatial coordinates.

Standalone (evaluate cells you already generated)::

    from paired_slides_eval import TargetSlide, GeneratedSlide, GeneratedNiches, evaluate

    target = TargetSlide.from_anndata("target.h5ad", ct_key="class")   # raw genes + obsm['spatial']

    # whole-slide model -> flat cells (label-free metrics; niche metrics skipped):
    generated = GeneratedSlide.from_anndata("generated.h5ad")          # (N, D)
    # OR NicheFlow-style microenvironments (enables the niche metrics too):
    generated = GeneratedNiches.from_anndata("generated.h5ad")         # (B, N, D), centroid first

    results = evaluate(target, generated)                              # {test/group/metric: value}

Full pipeline (checkpoint + raw slides -> generate -> metrics). The generation step is a
pluggable blackbox: pass any ``generator`` implementing the
:class:`~paired_slides_eval.pipeline.Generator` protocol.
The bundled NicheFlow adapter (needs the ``[pipeline]`` extra) is one such generator::

    from paired_slides_eval.pipeline import run_pipeline
    from paired_slides_eval.adapters.nicheflow import nicheflow_generator

    res = run_pipeline(
        "source.h5ad", "target.h5ad", "flow.ckpt",
        generator=nicheflow_generator, classifier_h5ad="clf.h5ad",
    )

Bring your own model: write a ``generator`` that returns a
:class:`~paired_slides_eval.pipeline.GenerationOutput` (``from_generated_anndata`` does this in
one line from a generated ``.h5ad``) — no NicheFlow needed.

Generate and evaluate as **separate steps** (generate once, evaluate many times) via the
model-agnostic generate entry point :mod:`paired_slides_eval.generate`::

    python -m paired_slides_eval.generate --generator mypkg.mymodel:my_generator \\
        --source source.h5ad --target target.h5ad --checkpoint flow.ckpt --generated_out gen.h5ad
    python -m paired_slides_eval.evaluate --target target.h5ad --generated gen.h5ad
"""

from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide, TargetSlide
from paired_slides_eval.evaluate import ALL_GROUPS, evaluate
from paired_slides_eval.generate import generate_cells, write_generated

__all__ = [
    "ALL_GROUPS",
    "GeneratedNiches",
    "GeneratedSlide",
    "TargetSlide",
    "evaluate",
    "generate_cells",
    "write_generated",
]
