"""paired-slides-eval — evaluation metrics for spatial single-cell generation.

This is an **evaluation** library. It is model-agnostic: you bring a real **target slide** and the
**generated cells** your model produced (both as AnnData / arrays), and it computes the metric
suite. It knows nothing about how you generated — generation lives in *your* code, where your model
is. (An optional integrated-generation layer exists for models that ship an adapter; see below.)

The one-call front door — evaluate two files (do this right after your own pipeline samples)::

    from paired_slides_eval import evaluate_files

    metrics = evaluate_files(
        "target.h5ad",        # real slide: raw genes + obsm['spatial']
        "generated.h5ad",     # your model's output (flat X+coords, or niche-shaped)
        ct_key="class",       # enables the classifier metrics (optional)
        n_pcs=50,             # shared PCA so both sides live in one space
    )                         # -> {test/group/metric: value}

Or build the inputs explicitly and call :func:`~paired_slides_eval.evaluate.evaluate`::

    from paired_slides_eval import TargetSlide, GeneratedSlide, evaluate

    target = TargetSlide.from_anndata("target.h5ad", ct_key="class", n_pcs=50)
    generated = GeneratedSlide.from_anndata("generated.h5ad").project(target.pca)
    metrics = evaluate(target, generated)

Generated cells come in two shapes: a flat :class:`GeneratedSlide` (``X`` + coords, for whole-slide
models) or a niche-shaped :class:`GeneratedNiches` (``obs['niche_id']``, for microenvironment
models). The label-free metrics run on either; see :func:`~paired_slides_eval.evaluate.evaluate`.

Optional — integrated generation. If a model ships a ``generator`` adapter you can also run
generate + evaluate from here; this is a convenience, not the core. See
:mod:`paired_slides_eval.pipeline` (``run_pipeline``) and :mod:`paired_slides_eval.generate`
(the standalone generate CLI). The bundled NicheFlow adapter (``[nicheflow]`` extra) is one such
generator; bring your own by implementing the
:class:`~paired_slides_eval.pipeline.Generator` protocol.
"""

from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide, TargetSlide
from paired_slides_eval.evaluate import ALL_GROUPS, evaluate, evaluate_files
from paired_slides_eval.generate import generate_cells, write_generated

__all__ = [
    "ALL_GROUPS",
    "GeneratedNiches",
    "GeneratedSlide",
    "TargetSlide",
    "evaluate",
    "evaluate_files",
    "generate_cells",
    "write_generated",
]
