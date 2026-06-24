import numpy as np

from paired_slides_eval.metrics.distances import point_to_shape, regression_metrics, shape_to_point


def test_psd_zero_when_generated_equals_real(rng):
    pos = rng.uniform(0, 10, size=(200, 2))
    out = point_to_shape(pos, pos, prefix="test")
    assert out["test/psd/mean"] == 0.0
    assert out["test/psd/max"] == 0.0


def test_spd_keys_and_positive(rng):
    real = rng.uniform(0, 10, size=(200, 2))
    gen = rng.uniform(0, 10, size=(150, 2))
    out = shape_to_point(gen, real, prefix="test")
    assert set(out) == {"test/spd/mean", "test/spd/max"}
    assert out["test/spd/mean"] >= 0.0


def test_regression_zero_on_identical(rng):
    x = rng.normal(size=(30, 6, 5))
    pos = rng.normal(size=(30, 6, 2))
    out = regression_metrics(x, x, pos, pos, prefix="test")
    assert out == {"test/x/mse": 0.0, "test/x/mae": 0.0, "test/pos/mse": 0.0, "test/pos/mae": 0.0}


def test_regression_increases_with_error(rng):
    x = rng.normal(size=(30, 6, 5))
    pos = rng.normal(size=(30, 6, 2))
    out = regression_metrics(x + 2.0, x, pos, pos, prefix="test")
    assert out["test/x/mae"] > 1.0
