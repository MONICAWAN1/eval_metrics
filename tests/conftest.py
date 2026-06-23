"""Synthetic fixtures so the metric tests run without any real data / checkpoints."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def real_slide(rng):
    """A small fake target slide: clustered PCA expression + 2-D coords + integer cell types."""
    n, n_pcs, n_classes = 400, 8, 4
    pos = rng.uniform(0, 10, size=(n, 2))
    # expression weakly correlated with position so Moran's I is non-trivial
    x = rng.normal(size=(n, n_pcs)) + 0.3 * pos[:, :1]
    ct = rng.integers(0, n_classes, size=n)
    return {"x": x, "pos": pos, "ct": ct, "n_classes": n_classes}


@pytest.fixture
def generated_niches(real_slide, rng):
    """(B, N, D) generated niches drawn near the real slide (centroid at index 0)."""
    b, n_points = 60, 6
    n_pcs = real_slide["x"].shape[1]
    centroids = rng.uniform(1, 9, size=(b, 2))
    pos = centroids[:, None, :] + rng.normal(scale=0.3, size=(b, n_points, 2))
    x = rng.normal(size=(b, n_points, n_pcs)) + 0.3 * pos[:, :, :1]
    return {"x": x, "pos": pos}
