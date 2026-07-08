from paired_slides_eval.metrics.distances import regression_metrics


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
