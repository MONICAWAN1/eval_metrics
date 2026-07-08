"""Paired-slides-eval — evaluation metrics for spatial single-cell generation.

A model-agnostic evaluation library. It scores a set of generated cells against a real target slide
and reports the full metric suite. Evaluation is independent of generation; an optional
configuration-driven generation layer is available for models that ship an adapter (see below).

The one-call entry point evaluates two files::

    from paired_slides_eval import evaluate_files

    metrics = evaluate_files(
        "target.h5ad",        # real slide: raw genes + obsm['spatial']
        "generated.h5ad",     # generated cells (flat X+coords, or niche-shaped)
        ct_key="class",       # enables the classifier metrics (optional)
        n_pcs=50,             # shared PCA so both sides live in one space
    )                         # -> {test/group/metric: value}

The inputs can also be built explicitly and passed to
:func:`~paired_slides_eval.evaluate.evaluate`::

    from paired_slides_eval import TargetSlide, GeneratedSlide, evaluate

    target = TargetSlide.from_anndata("target.h5ad", ct_key="class", n_pcs=50)
    generated = GeneratedSlide.from_anndata("generated.h5ad").project(target.pca)
    metrics = evaluate(target, generated)

Generated cells take one of two shapes: a flat :class:`GeneratedSlide` (``X`` + coordinates, for
whole-slide models) or a niche-shaped :class:`GeneratedNiches` (``obs['niche_id']``, for
microenvironment models). The label-free metrics run on either.

Optional generation. Models with an adapter under :mod:`paired_slides_eval.adapters` can be run from
configuration via :mod:`paired_slides_eval.generate` and :mod:`paired_slides_eval.pipeline`
(``run_pipeline``); these require the ``[pipeline]`` extra. The bundled NicheFlow adapter
(``NicheFlowGenerator``) is one such adapter.

"""

from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide, TargetSlide
from paired_slides_eval.evaluate import ALL_GROUPS, evaluate, evaluate_files
from paired_slides_eval.pipeline import generate_cells, write_generated

__all__ = [
    "ALL_GROUPS",
    "GeneratedNiches",
    "GeneratedSlide",
    "TargetSlide",
    "evaluate",
    "evaluate_files",
    "generate_cells",
    "write_generated",
    "me",
    "pp",
]


def __getattr__(name):
    """Lazy namespaces: ``me`` (metrics + table wrapper), ``pp`` (preprocessing).

    Kept lazy so ``import paired_slides_eval`` stays lightweight — ``pp`` only pulls scanpy/torch
    when first accessed, and ``me`` only the metric stack.
    """
    if name == "me":
        import paired_slides_eval.metrics as me

        return me
    if name == "pp":
        import paired_slides_eval.preprocessing as pp

        return pp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
