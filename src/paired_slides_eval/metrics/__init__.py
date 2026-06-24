"""Evaluation metrics for spatial single-cell generation.

Two layers per metric: the framework-free *kernel* (e.g. ``c2st``, ``mmd2_rbf``, ``morans_i``)
and a *wrapper* that scores generated-vs-real and returns a flat ``{group/metric: value}`` dict
(e.g. ``c2st_metrics``, ``distribution_distance``, ``morans_compare``). See
:func:`paired_slides_eval.evaluate.evaluate` to run the whole suite at once.
"""

from paired_slides_eval.metrics.c2st import c2st, c2st_metrics, c2st_significance
from paired_slides_eval.metrics.classifier_gap import classifier_accuracy_gap
from paired_slides_eval.metrics.concordance import cell_type_concordance
from paired_slides_eval.metrics.distances import (
    point_to_shape,
    regression_metrics,
    shape_to_point,
)
from paired_slides_eval.metrics.distribution import distribution_distance, mmd2_rbf, ot_distance
from paired_slides_eval.metrics.morans import morans_compare, morans_i

__all__ = [
    # kernels
    "c2st",
    "c2st_significance",
    "mmd2_rbf",
    "ot_distance",
    "morans_i",
    # wrappers
    "c2st_metrics",
    "cell_type_concordance",
    "classifier_accuracy_gap",
    "distribution_distance",
    "morans_compare",
    "point_to_shape",
    "regression_metrics",
    "shape_to_point",
]
