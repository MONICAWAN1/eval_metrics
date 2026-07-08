"""The scib-style DataFrame table front door (`metrics.metrics` / `compare`).

Pure numpy — uses the `regression` group so no torch/sklearn is needed.

"""

import numpy as np
import pandas as pd

from paired_slides_eval import GeneratedNiches, TargetSlide, evaluate
from paired_slides_eval.metrics import compare, metrics

GROUPS = ("regression",)


def _pair(seed):
    rng = np.random.default_rng(seed)
    x = rng.random((6, 4, 5))
    pos = rng.random((6, 4, 2))
    return (
        TargetSlide(x=rng.random((50, 5)), pos=rng.random((50, 2))),
        GeneratedNiches(x=x, pos=pos, gt_x=x.copy(), gt_pos=pos.copy()),
    )


def test_metrics_returns_tidy_frame():
    t, g = _pair(0)
    df = metrics(t, g, name="m", groups=GROUPS)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "metric" and list(df.columns) == ["m"]
    assert {"test/x/mse", "test/x/mae", "test/pos/mse", "test/pos/mae"} <= set(df.index)
    assert "skipped" in df.attrs and "notes" in df.attrs
    assert not any(k.startswith("_") for k in df.index)  # private keys dropped


def test_metrics_matches_evaluate():
    t, g = _pair(1)
    df = metrics(t, g, groups=GROUPS)
    flat = evaluate(t, g, groups=GROUPS)
    for k, v in flat.items():
        if not k.startswith("_"):
            assert df.loc[k, "value"] == v  # identical numbers, just reshaped


def test_compare_wide_table():
    t, g1 = _pair(2)
    _, g2 = _pair(3)
    wide = compare({"a": (t, g1), "b": (t, g2)}, groups=GROUPS)
    assert list(wide.columns) == ["a", "b"]
    assert wide.index.name == "metric" and wide.shape[0] >= 4
