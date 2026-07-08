"""Evaluation metrics for spatial single-cell generation (the ``me`` namespace).

Three layers: the framework-free *kernel* (e.g. ``c2st``, ``mmd2_rbf``, ``morans_i``); a *wrapper*
that scores generated-vs-real and returns a flat ``{group/metric: value}`` dict (e.g. ``c2st_metrics``,
``distribution_distance``, ``morans_compare``); and the *table* front door — :func:`metrics` /
:func:`metrics_files` / :func:`compare` — returning a tidy ``pandas.DataFrame``. See
:func:`paired_slides_eval.evaluate.evaluate` for the underlying suite.

"""

from paired_slides_eval.metrics.c2st import c2st, c2st_metrics, c2st_significance
from paired_slides_eval.metrics.c2st_nn import c2st_nn, c2st_nn_metrics
from paired_slides_eval.metrics.classifier_gap import classifier_accuracy_gap
from paired_slides_eval.metrics.concordance import cell_type_concordance
from paired_slides_eval.metrics.distances import regression_metrics
from paired_slides_eval.metrics.distribution import distribution_distance, mmd2_rbf, ot_distance
from paired_slides_eval.metrics.expr_recon import expr_recon_gap, fixed_reference_mse
from paired_slides_eval.metrics.metrics import compare, metrics, metrics_files
from paired_slides_eval.metrics.morans import morans_compare, morans_i

__all__ = [
    # kernels
    "c2st",
    "c2st_nn",
    "c2st_significance",
    "mmd2_rbf",
    "ot_distance",
    "morans_i",
    # wrappers
    "c2st_metrics",
    "c2st_nn_metrics",
    "cell_type_concordance",
    "classifier_accuracy_gap",
    "distribution_distance",
    "expr_recon_gap",
    "fixed_reference_mse",
    "morans_compare",
    "regression_metrics",
    # table front door
    "metrics",
    "metrics_files",
    "compare",
]
