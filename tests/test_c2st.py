import numpy as np
import pytest

from paired_slides_eval.metrics.c2st import c2st, c2st_metrics


def test_c2st_chance_on_identical_distribution(rng):
    x = rng.normal(size=(300, 5))
    y = rng.normal(size=(300, 5))
    acc, auc = c2st(x, y, seed=0, n_folds=3)
    assert 0.4 < acc < 0.65
    assert 0.4 < auc < 0.65


def test_c2st_separates_shifted_distribution(rng):
    x = rng.normal(size=(300, 5))
    y = rng.normal(size=(300, 5)) + 5.0
    acc, _ = c2st(x, y, seed=0, n_folds=3)
    assert acc > 0.9


def test_c2st_metrics_keys(real_slide, generated_niches):
    pytest.importorskip("sklearn")
    gen_x = generated_niches["x"].reshape(-1, generated_niches["x"].shape[-1])
    gen_pos = generated_niches["pos"].reshape(-1, 2)
    out = c2st_metrics(
        real_slide["x"],
        real_slide["pos"],
        gen_x,
        gen_pos,
        prefix="test",
        n_folds=3,
    )
    assert set(out) == {
        "test/c2st/acc",
        "test/c2st/auc",
        "test/c2st/gene_acc",
        "test/c2st/gene_auc",
    }
    assert all(np.isfinite(v) for v in out.values())
