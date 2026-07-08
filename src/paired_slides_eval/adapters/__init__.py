"""Generation adapters.

Each adapter wraps one generative model behind the
:class:`~paired_slides_eval.adapters.base.BaseGenerator` contract, isolating that model's imports and
preprocessing. Adapters are selected and constructed from a Hydra config via ``_target_``
(``configs/generator/<name>.yaml``). The bundled adapter is
:mod:`paired_slides_eval.adapters.nicheflow` (``NicheFlowGenerator``).

To add a model: place an adapter subpackage here with a ``BaseGenerator`` subclass that imports the
model and returns a :class:`~paired_slides_eval.pipeline.run.GenerationOutput`, then add a matching
``configs/generator/<name>.yaml`` pointing ``_target_`` at the class.

"""
