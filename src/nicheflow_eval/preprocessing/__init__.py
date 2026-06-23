"""Niche preprocessing: raw AnnData slides -> the niche dataclass (radius graph + grid subsample).

A faithful port of NicheFlow's preprocessing (``scripts/prepare_abca`` +
``nicheflow.preprocessing.h5ad_preprocessor``), kept in this standalone repo so the niche-based
metrics need only raw ``.h5ad`` inputs — no preprocessed pickle. Global alignment (PASTE2) is
omitted; coordinates are standardized per slide.
"""

from nicheflow_eval.preprocessing.h5ad_preprocessor import H5ADPreprocessor
from nicheflow_eval.preprocessing.prepare import (
    compute_pca,
    preprocess_classifier_slide,
    preprocess_pair,
)

__all__ = [
    "H5ADPreprocessor",
    "compute_pca",
    "preprocess_pair",
    "preprocess_classifier_slide",
]
