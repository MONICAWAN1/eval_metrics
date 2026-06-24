"""Model-agnostic checkpoint -> generated cells -> metrics pipeline.

This package imports **no** generative model. :func:`run_pipeline` takes a ``generator`` — any
callable implementing the :class:`Generator` contract — so you can plug in your own model. The
bundled NicheFlow implementation lives separately in
:mod:`nicheflow_eval.adapters.nicheflow` (needs the ``[pipeline]`` extra).

To evaluate cells you generated elsewhere, skip this and use
:func:`nicheflow_eval.evaluate.evaluate` directly.
"""

from nicheflow_eval.pipeline.run import (
    GenerationOutput,
    Generator,
    PipelineResult,
    from_generated_anndata,
    run_pipeline,
)

__all__ = [
    "Generator",
    "GenerationOutput",
    "PipelineResult",
    "from_generated_anndata",
    "run_pipeline",
]
