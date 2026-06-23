"""Evaluation metrics for spatial single-cell generation.

Two layers per metric: the framework-free *kernel* (e.g. ``c2st``, ``mmd2_rbf``, ``morans_i``)
and a *wrapper* that scores generated-vs-real and returns a flat ``{group/metric: value}`` dict
(e.g. ``c2st_metrics``, ``distribution_distance``, ``morans_compare``). See
:func:`nicheflow_eval.evaluate.evaluate` to run the whole suite at once.
"""

from nicheflow_eval.metrics.c2st import c2st, c2st_metrics, c2st_significance
from nicheflow_eval.metrics.concordance import cell_type_concordance
from nicheflow_eval.metrics.distances import (
    point_to_shape,
    regression_metrics,
    shape_to_point,
)
from nicheflow_eval.metrics.distribution import distribution_distance, mmd2_rbf, ot_distance
from nicheflow_eval.metrics.morans import morans_compare, morans_i

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
    "distribution_distance",
    "morans_compare",
    "point_to_shape",
    "regression_metrics",
    "shape_to_point",
]
