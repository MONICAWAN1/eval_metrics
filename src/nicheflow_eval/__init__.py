"""nicheflow-eval — standalone evaluation metrics for spatial single-cell generation.

Quick start::

    from nicheflow_eval import TargetSlide, GeneratedNiches, evaluate
    from nicheflow_eval.data import load_h5ad_dataset_dataclass

    ds = load_h5ad_dataset_dataclass("data/abca.pkl")     # reuse the preprocessing .pkl
    target = TargetSlide.from_dataclass(ds, timepoint=ds.timepoints_ordered[-1])
    generated = GeneratedNiches(x=gen_x, pos=gen_pos)      # (B, N, D), centroid at index 0
    results = evaluate(target, generated)                 # {test/group/metric: value}
"""

from nicheflow_eval.contract import GeneratedNiches, TargetSlide
from nicheflow_eval.evaluate import ALL_GROUPS, evaluate

__all__ = ["ALL_GROUPS", "GeneratedNiches", "TargetSlide", "evaluate"]
