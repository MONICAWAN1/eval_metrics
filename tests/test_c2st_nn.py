"""Spatially-pooled nearest-neighbour two-sample test (c2st_nn). Pure numpy/scipy."""

from paired_slides_eval.metrics.c2st_nn import c2st_nn_metrics

KEYS = {"test/c2st/nn", "test/c2st/nn_std", "test/c2st/nn_real_ref"}


def test_c2st_nn_chance_on_identical(rng):
    real_x = rng.normal(size=(400, 5))
    gen_x = rng.normal(size=(400, 5))  # same distribution
    real_pos = rng.uniform(0, 20, size=(400, 2))
    gen_pos = rng.uniform(0, 20, size=(400, 2))
    out = c2st_nn_metrics(real_x, real_pos, gen_x, gen_pos, prefix="test", spatial_k=8, seed=0)
    assert set(out) == KEYS
    assert 0.4 <= out["test/c2st/nn"] <= 0.6           # indistinguishable -> chance
    assert 0.4 <= out["test/c2st/nn_real_ref"] <= 0.6  # self-calibration also ~0.5


def test_c2st_nn_low_when_separable(rng):
    real_x = rng.normal(size=(400, 5))
    gen_x = rng.normal(size=(400, 5)) + 12.0  # far from real -> gen NNs are other gen cells
    real_pos = rng.uniform(0, 20, size=(400, 2))
    gen_pos = rng.uniform(0, 20, size=(400, 2))
    out = c2st_nn_metrics(real_x, real_pos, gen_x, gen_pos, prefix="test", spatial_k=8, seed=0)
    assert out["test/c2st/nn"] < 0.1


def test_c2st_nn_high_when_memorized(rng):
    real_x = rng.normal(size=(400, 5))
    real_pos = rng.uniform(0, 20, size=(400, 2))
    out = c2st_nn_metrics(  # generated == real copies -> each gen NN is its real twin
        real_x, real_pos, real_x.copy(), real_pos.copy(), prefix="test", spatial_k=8, seed=0
    )
    assert out["test/c2st/nn"] > 0.9
