"""Model-specific generation adapters that plug into the model-agnostic pipeline.

Each adapter implements the :class:`~nicheflow_eval.pipeline.run.Generator` contract for one
generative model, isolating its imports and preprocessing here. The bundled one is
:mod:`nicheflow_eval.adapters.nicheflow`; add your own by writing a callable with the same
signature.
"""
