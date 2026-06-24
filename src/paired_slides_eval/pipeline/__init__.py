"""Model-agnostic checkpoint -> generated cells -> metrics pipeline.

This package imports **no** generative model. :func:`run_pipeline` takes a ``generator`` — any
callable implementing the :class:`Generator` contract — so you can plug in your own model. The
bundled NicheFlow implementation lives separately in
:mod:`paired_slides_eval.adapters.nicheflow` (needs the ``[pipeline]`` extra).

To evaluate cells you generated elsewhere, skip this and use
:func:`paired_slides_eval.evaluate.evaluate` directly.
"""

from paired_slides_eval.pipeline.run import (
    GENERATOR_REGISTRY,
    GenerationOutput,
    Generator,
    PipelineResult,
    from_generated_anndata,
    resolve_generator,
    run_pipeline,
)

__all__ = [
    "GENERATOR_REGISTRY",
    "Generator",
    "GenerationOutput",
    "PipelineResult",
    "from_generated_anndata",
    "resolve_generator",
    "run_pipeline",
]
