"""The full checkpoint -> generated cells -> metrics pipeline (imports ``nicheflow`` as a blackbox).

Use :func:`run_pipeline` for the end-to-end flow, or :func:`generate` to just produce generated
cells from a checkpoint. For evaluating externally-generated cells, use
:func:`nicheflow_eval.evaluate.evaluate` instead (no ``nicheflow`` needed).
"""

from nicheflow_eval.pipeline.generate import GenerationResult, generate
from nicheflow_eval.pipeline.run import PipelineResult, run_pipeline

__all__ = ["GenerationResult", "PipelineResult", "generate", "run_pipeline"]
