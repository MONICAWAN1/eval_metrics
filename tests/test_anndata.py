"""Tests for the AnnData input path: TargetSlide/GeneratedNiches.from_anndata + shared PCA."""

import numpy as np
import pytest

from nicheflow_eval import GeneratedNiches, TargetSlide


def test_target_from_anndata_raw_genes(target_adata):
    t = TargetSlide.from_anndata(target_adata, ct_key="class")
    assert t.x.shape == (target_adata.n_obs, target_adata.n_vars)
    assert t.pos.shape == (target_adata.n_obs, 2)
    assert t.ct is not None and t.ct.shape == (target_adata.n_obs,)
    assert t.n_classes == 4
    assert t.pca is None  # no PCA requested -> raw-gene space


def test_target_from_anndata_with_pca(target_adata):
    t = TargetSlide.from_anndata(target_adata, ct_key="class", n_pcs=10)
    assert t.x.shape == (target_adata.n_obs, 10)
    assert t.pca is not None


def test_generated_from_anndata_roundtrip(generated_adata, generated_niches):
    g = GeneratedNiches.from_anndata(generated_adata)
    b, n, d = generated_niches["x"].shape
    assert g.x.shape == (b, n, d)
    assert g.pos.shape == (b, n, 2)
    assert g.gt_x.shape == (b, n, d)
    assert g.gt_ct is not None and g.gt_ct.shape == (b,)
    # niche grouping preserved the per-cell values
    np.testing.assert_allclose(g.flat_x.sum(), generated_niches["x"].sum(), rtol=1e-5)


def test_project_shares_basis(target_adata, rng):
    # Generated cells share the target's raw-gene space (same n_vars), then both project to PCA.
    t = TargetSlide.from_anndata(target_adata, n_pcs=10)
    g_genes = target_adata.n_vars
    raw = GeneratedNiches(
        x=rng.normal(size=(8, 6, g_genes)).astype("float32"),
        pos=rng.uniform(0, 5, size=(8, 6, 2)).astype("float32"),
        gt_x=rng.normal(size=(8, 6, g_genes)).astype("float32"),
        gt_pos=rng.uniform(0, 5, size=(8, 6, 2)).astype("float32"),
    )
    g = raw.project(t.pca)
    assert g.x.shape[-1] == 10
    assert g.gt_x.shape[-1] == 10
    # project is a no-op when pca is None
    assert raw.project(None).x.shape[-1] == g_genes


def test_project_none_is_noop_identity(generated_adata):
    g = GeneratedNiches.from_anndata(generated_adata)
    assert g.project(None) is g


def test_generated_requires_uniform_niche_size(target_adata):
    ad = pytest.importorskip("anndata")
    import pandas as pd

    x = np.zeros((5, 3), dtype="float32")
    adata = ad.AnnData(X=x)
    adata.obs["niche_id"] = np.array([0, 0, 1, 1, 1])  # uneven sizes
    adata.obsm["spatial"] = np.zeros((5, 2), dtype="float32")
    adata.obs.index = pd.RangeIndex(5).astype(str)
    with pytest.raises(ValueError):
        GeneratedNiches.from_anndata(adata)
