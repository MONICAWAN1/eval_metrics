"""Tests for the flat GeneratedSlide path: whole-slide models, niche metrics auto-skipped."""

import numpy as np
import pytest

from paired_slides_eval import GeneratedNiches, GeneratedSlide, TargetSlide, evaluate


def test_niches_to_slide_is_single_flatten_path(generated_adata):
    niches = GeneratedNiches.from_anndata(generated_adata)
    slide = niches.to_slide()
    assert isinstance(slide, GeneratedSlide)
    b, n, d = niches.x.shape
    assert slide.x.shape == (b * n, d)
    # flat_x / flat_pos are views over to_slide()
    np.testing.assert_array_equal(niches.flat_x, slide.x)
    np.testing.assert_array_equal(niches.flat_pos, slide.pos)
    # a flat slide's to_slide() is itself
    assert slide.to_slide() is slide


def test_generated_slide_from_anndata(generated_slide_adata):
    g = GeneratedSlide.from_anndata(generated_slide_adata)
    n, d = generated_slide_adata.n_obs, generated_slide_adata.n_vars
    assert g.x.shape == (n, d)
    assert g.pos.shape == (n, 2)
    # flat_x / flat_pos are the arrays themselves (same interface as GeneratedNiches)
    assert g.flat_x is g.x and g.flat_pos is g.pos


def test_generated_slide_rejects_niche_shape(rng):
    with pytest.raises(ValueError):
        GeneratedSlide(x=rng.normal(size=(4, 6, 8)), pos=rng.normal(size=(4, 6, 2)))


def test_generated_slide_project_shares_basis(target_adata, generated_slide_adata):
    t = TargetSlide.from_anndata(target_adata, n_pcs=10)
    g = GeneratedSlide.from_anndata(generated_slide_adata).project(t.pca)
    assert g.x.shape[-1] == 10
    assert g.pos.shape[-1] == 2  # coordinates untouched
    # no-op when pca is None
    raw = GeneratedSlide.from_anndata(generated_slide_adata)
    assert raw.project(None) is raw


def test_evaluate_flat_slide_runs_label_free_and_skips_niche(target_adata, generated_slide_adata):
    target = TargetSlide.from_anndata(target_adata, ct_key="class")
    gen = GeneratedSlide.from_anndata(generated_slide_adata)

    res = evaluate(target, gen, groups=("psd", "spd", "regression", "ct_gap"))

    # label-free groups ran on the flat cloud
    assert any(k.startswith("test/psd") for k in res)
    assert any(k.startswith("test/spd") for k in res)
    # niche groups skipped (regression needs gt_*; ct_gap needs niches + classifier)
    skipped = res["_skipped"]
    assert any("regression" in s for s in skipped)
    assert any("ct_gap" in s for s in skipped)


def test_load_generated_autodetects_flat(tmp_path, rng):
    from paired_slides_eval.evaluate import _load_generated

    # flat .npz (2-D x) -> GeneratedSlide
    npz_path = tmp_path / "flat.npz"
    np.savez(npz_path, x=rng.normal(size=(30, 8)), pos=rng.uniform(0, 5, size=(30, 2)))
    g = _load_generated(str(npz_path))
    assert isinstance(g, GeneratedSlide)
    assert g.x.shape == (30, 8)
