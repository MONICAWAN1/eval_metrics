"""Feature-space auto-detection: project gene-space cells, pass through already-PCA cells."""

import numpy as np
import pytest

from paired_slides_eval import GeneratedNiches, GeneratedSlide, TargetSlide
from paired_slides_eval.data.anndata import _PCA
from paired_slides_eval.pipeline import from_generated_arrays

N_GENES, N_PCS = 20, 5


@pytest.fixture
def pca(rng):
    """A frozen PCA mapping N_GENES raw features -> N_PCS components."""
    return _PCA(mean=rng.random(N_GENES), components=rng.random((N_PCS, N_GENES)))


def test_project_gene_space_is_reduced(pca, rng):
    gen = GeneratedSlide(x=rng.random((30, N_GENES)), pos=rng.random((30, 2)))
    assert gen.project(pca).x.shape == (30, N_PCS)  # gene space -> projected


def test_project_already_pca_is_passthrough(pca, rng):
    gen = GeneratedSlide(x=rng.random((30, N_PCS)), pos=rng.random((30, 2)))
    out = gen.project(pca)
    assert out.x.shape == (30, N_PCS)
    assert np.array_equal(out.x, gen.x)  # already reduced -> untouched, NOT double-transformed


def test_project_dimension_mismatch_raises(pca, rng):
    gen = GeneratedSlide(x=rng.random((30, 7)), pos=rng.random((30, 2)))
    with pytest.raises(ValueError, match="match neither"):
        gen.project(pca)


def test_project_none_is_noop(rng):
    gen = GeneratedSlide(x=rng.random((30, N_PCS)), pos=rng.random((30, 2)))
    assert gen.project(None) is gen


def test_niches_project_detects_and_keeps_gt(pca, rng):
    # already-PCA niches (incl. gt_x) -> all passed through unchanged
    gn = GeneratedNiches(
        x=rng.random((4, 6, N_PCS)),
        pos=rng.random((4, 6, 2)),
        gt_x=rng.random((4, 6, N_PCS)),
        gt_pos=rng.random((4, 6, 2)),
    )
    out = gn.project(pca)
    assert np.array_equal(out.x, gn.x) and np.array_equal(out.gt_x, gn.gt_x)
    # gene-space niches -> projected
    gene = GeneratedNiches(x=rng.random((4, 6, N_GENES)), pos=rng.random((4, 6, 2)))
    assert gene.project(pca).x.shape == (4, 6, N_PCS)


def test_from_generated_arrays_prebuilt_target_already_pca(pca, rng):
    # target already in PCA space (pca=None) + already-PCA generated -> no projection
    target = TargetSlide(
        x=rng.random((50, N_PCS)),
        pos=rng.random((50, 2)),
        ct=rng.integers(0, 3, 50),
        n_classes=3,
        pca=None,
    )
    out = from_generated_arrays(rng.random((40, N_PCS)), rng.random((40, 2)), target)
    assert out.target is target
    assert isinstance(out.generated, GeneratedSlide) and out.generated.x.shape == (40, N_PCS)


def test_from_generated_arrays_gene_space_projected(pca, rng):
    target = TargetSlide(
        x=rng.random((50, N_PCS)),
        pos=rng.random((50, 2)),
        ct=None,
        n_classes=None,
        pca=pca,
    )
    out = from_generated_arrays(rng.random((40, N_GENES)), rng.random((40, 2)), target)
    assert out.generated.x.shape == (
        40,
        N_PCS,
    )  # gene-space arrays projected through the target PCA


def test_from_generated_arrays_niche_shaped(rng):
    target = TargetSlide(
        x=rng.random((50, N_PCS)),
        pos=rng.random((50, 2)),
        ct=rng.integers(0, 3, 50),
        n_classes=3,
        pca=None,
    )
    out = from_generated_arrays(
        rng.random((4, 6, N_PCS)),
        rng.random((4, 6, 2)),
        target,
        gt_ct=np.arange(4),
    )
    assert isinstance(out.generated, GeneratedNiches) and out.generated.x.shape == (4, 6, N_PCS)
