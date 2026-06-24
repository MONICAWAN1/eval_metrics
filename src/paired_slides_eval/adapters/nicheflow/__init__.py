"""NicheFlow adapter — the reference implementation of the generation blackbox.

``nicheflow_generator`` plugs into :func:`paired_slides_eval.pipeline.run.run_pipeline` as
``generator=...``. It is the only part of the package that imports the ``nicheflow`` flow model
(plus ``torch`` / ``scanpy`` for preprocessing), so the standalone path stays dependency-light.

Needs the ``[pipeline]`` extra: ``pip install -e ../nicheflow_mba``.

Lower-level pieces (``preprocess_pair``, ``generate``, the ``H5ADPreprocessor``) are exported too,
for users who want to drive the steps by hand.
"""

from paired_slides_eval.adapters.nicheflow.generate import GenerationResult, generate
from paired_slides_eval.adapters.nicheflow.generator import (
    NicheFlowGenerator,
    nicheflow_generator,
    target_from_dataclass,
)
from paired_slides_eval.adapters.nicheflow.h5ad_preprocessor import H5ADPreprocessor
from paired_slides_eval.adapters.nicheflow.preprocess import (
    compute_pca,
    preprocess_classifier_slide,
    preprocess_pair,
)

__all__ = [
    "NicheFlowGenerator",
    "nicheflow_generator",
    "target_from_dataclass",
    "generate",
    "GenerationResult",
    "H5ADPreprocessor",
    "compute_pca",
    "preprocess_pair",
    "preprocess_classifier_slide",
]
