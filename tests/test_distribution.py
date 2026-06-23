import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("ot")

from nicheflow_eval.metrics.distribution import distribution_distance, mmd2_rbf, ot_distance


def test_mmd_near_zero_for_same_distribution(rng):
    x = rng.normal(size=(500, 4))
    y = rng.normal(size=(500, 4))
    assert abs(mmd2_rbf(x, y, seed=0)) < 0.05


def test_mmd_grows_with_shift(rng):
    x = rng.normal(size=(500, 4))
    near = rng.normal(size=(500, 4)) + 0.5
    far = rng.normal(size=(500, 4)) + 5.0
    assert mmd2_rbf(x, far, seed=0) > mmd2_rbf(x, near, seed=0)


def test_wasserstein_monotone_in_shift(rng):
    x = rng.normal(size=(400, 3))
    near = rng.normal(size=(400, 3)) + 1.0
    far = rng.normal(size=(400, 3)) + 4.0
    assert ot_distance(x, far, power=2, seed=0) > ot_distance(x, near, power=2, seed=0)


def test_distribution_distance_keys(rng):
    real_x, real_pos = rng.normal(size=(300, 5)), rng.normal(size=(300, 2))
    gen_x, gen_pos = rng.normal(size=(280, 5)), rng.normal(size=(280, 2))
    out = distribution_distance(real_x, real_pos, gen_x, gen_pos, prefix="test")
    assert set(out) == {
        "test/mmd2/x",
        "test/mmd2/pos",
        "test/ot_w1/x",
        "test/ot_w1/pos",
        "test/ot_w2/x",
        "test/ot_w2/pos",
    }
