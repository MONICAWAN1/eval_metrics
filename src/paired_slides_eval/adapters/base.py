"""Generation-adapter contract.

A generation adapter turns raw source/target slides plus a trained checkpoint into a comparable
``(target, generated)`` pair — a :class:`~paired_slides_eval.pipeline.run.GenerationOutput`. Adapters
subclass :class:`BaseGenerator`, live under ``paired_slides_eval.adapters``, and are selected and
constructed from a Hydra config via ``_target_``. The constructed object is callable and satisfies
the :class:`~paired_slides_eval.pipeline.run.Generator` protocol, so it can also be passed directly
to :func:`~paired_slides_eval.pipeline.run.run_pipeline`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from paired_slides_eval.pipeline.run import GenerationOutput


class BaseGenerator(ABC):
    """Base class for generation adapters.

    Model-specific parameters (PCA dimension, sampler settings, …) are constructor arguments and are
    supplied by a Hydra config. The call performs generation for a single ``(source, target,
    checkpoint)`` and returns the comparable pair.
    """

    @abstractmethod
    def __call__(self, *, source, target, checkpoint, **kwargs) -> GenerationOutput:
        """Generate cells for one ``(source, target, checkpoint)`` and return a ``GenerationOutput``."""
        raise NotImplementedError
