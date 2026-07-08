"""Synthetic fixtures so the metric tests run without any real data /
checkpoints."""

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
    """(B, N, D) generated niches drawn near the real slide (centroid at index
    0)."""
    b, n_points = 60, 6
    n_pcs = real_slide["x"].shape[1]
    centroids = rng.uniform(1, 9, size=(b, 2))
    pos = centroids[:, None, :] + rng.normal(scale=0.3, size=(b, n_points, 2))
    x = rng.normal(size=(b, n_points, n_pcs)) + 0.3 * pos[:, :, :1]
    return {"x": x, "pos": pos}


@pytest.fixture
def target_adata(rng):
    """A synthetic raw-gene target slide as AnnData: .X genes, obsm['spatial'], obs['class']."""
    ad = pytest.importorskip("anndata")
    import pandas as pd

    n, n_genes, n_classes = 300, 20, 4
    pos = rng.uniform(0, 10, size=(n, 2)).astype("float32")
    x = (rng.normal(size=(n, n_genes)) + 0.3 * pos[:, :1]).astype("float32")
    adata = ad.AnnData(X=x)
    adata.obsm["spatial"] = pos
    adata.obs["class"] = pd.Categorical(rng.choice(list("abcd")[:n_classes], size=n))
    return adata


@pytest.fixture
def generated_slide_adata(target_adata, rng):
    """A flat whole-slide generated AnnData: .X genes + obsm['spatial'], no niche_id."""
    ad = pytest.importorskip("anndata")
    import pandas as pd

    n_genes = target_adata.n_vars
    n = 250
    pos = rng.uniform(0, 10, size=(n, 2)).astype("float32")
    x = (rng.normal(size=(n, n_genes)) + 0.3 * pos[:, :1]).astype("float32")
    adata = ad.AnnData(X=x)
    adata.obsm["spatial"] = pos
    adata.obs.index = pd.RangeIndex(n).astype(str)
    return adata


@pytest.fixture
def generated_adata(generated_niches, rng):
    """A flat generated AnnData matching GeneratedNiches.from_anndata's
    layout."""
    ad = pytest.importorskip("anndata")
    import pandas as pd

    x, pos = generated_niches["x"], generated_niches["pos"]
    b, n, d = x.shape
    adata = ad.AnnData(X=x.reshape(b * n, d).astype("float32"))
    adata.obs["niche_id"] = np.repeat(np.arange(b), n)
    adata.obs["gt_ct"] = np.repeat(rng.integers(0, 4, size=b), n)
    adata.obsm["spatial"] = pos.reshape(b * n, 2).astype("float32")
    adata.obsm["gt_x"] = x.reshape(b * n, d).astype("float32")
    adata.obsm["gt_pos"] = pos.reshape(b * n, 2).astype("float32")
    adata.obs.index = pd.RangeIndex(b * n).astype(str)
    return adata
