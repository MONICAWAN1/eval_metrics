import numpy as np
import pytest

pytest.importorskip("squidpy")
pytest.importorskip("anndata")

from paired_slides_eval.metrics.morans import morans_compare, morans_i


def test_morans_i_detects_spatial_gradient(rng):
    pos = rng.uniform(0, 10, size=(300, 2))
    clustered = pos[:, :1]  # value == x-coordinate -> strongly autocorrelated
    noise = rng.normal(size=(300, 1))
    features = np.concatenate([clustered, noise], axis=1)
    i = morans_i(features, pos, n_neighs=6, seed=0)
    assert i[0] > i[1]  # gradient feature is more spatially clustered than noise


def test_morans_compare_keys(real_slide, generated_niches):
    gen_x = generated_niches["x"][:, 0, :]
    gen_pos = generated_niches["pos"][:, 0, :]
    out = morans_compare(
        gen_x,
        gen_pos,
        real_slide["x"],
        real_slide["pos"],
        prefix="test",
        n_neighs=6,
    )
    assert set(out) == {
        "test/moran/mae",
        "test/moran/corr",
        "test/moran/real_mean",
        "test/moran/gen_mean",
    }
