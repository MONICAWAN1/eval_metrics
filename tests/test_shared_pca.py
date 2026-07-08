"""Tests for the shared-PCA recipe transform, coord standardiser, and
comparability guards.

Pure numpy/sklearn — runs on the dev box (no torch). The fixed-centroid acc_real test needs torch
and is guarded by ``importorskip``.

"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA

from paired_slides_eval.data.shared_pca import (
    CoordStandardizer,
    SharedGenePCA,
    coord_standardizer_from_dataclass,
    shared_pca_from_dataclass,
)


def _fit_recipe(counts, n_pcs):
    """Mirror compute_pca + _normalize_features exactly, returning the pieces +
    whitened X_pca."""
    totals = counts.sum(1)
    target_sum = float(np.median(totals[totals > 0]))
    logn = np.log1p(counts / totals[:, None] * target_sum)
    pca = PCA(n_components=n_pcs).fit(logn)
    pcs = pca.components_.T  # scanpy's varm['PCs'] == components_.T  (n_genes, n_pcs)
    x_unwhit = (logn - pca.mean_) @ pcs
    xmean, xstd = x_unwhit.mean(0), x_unwhit.std(0)
    x_pca = (x_unwhit - xmean) / xstd
    spca = SharedGenePCA(
        pcs=pcs,
        lognorm_mean=pca.mean_,
        xpca_mean=xmean,
        xpca_std=xstd,
        target_sum=target_sum,
    )
    return spca, x_pca


def test_shared_gene_pca_reproduces_fit_projection():
    rng = np.random.default_rng(0)
    counts = rng.poisson(5, size=(60, 12)).astype(float)
    spca, x_pca = _fit_recipe(counts, n_pcs=4)
    got = spca.transform(counts)
    assert got.shape == (60, 4)
    assert np.allclose(got, x_pca, atol=1e-4)


def test_components_shape_enables_space_autodetect():
    # _pca_aware_transform keys off components: (n_pcs, n_genes) -> in_dim=n_genes, out_dim=n_pcs.
    from paired_slides_eval.contract import _pca_aware_transform

    rng = np.random.default_rng(1)
    counts = rng.poisson(5, size=(30, 10)).astype(float)
    spca, x_pca = _fit_recipe(counts, n_pcs=5)

    # gene-space (width == n_genes) gets projected...
    proj = _pca_aware_transform(counts, spca)
    assert np.allclose(proj, x_pca, atol=1e-4)
    # ...already-reduced (width == n_pcs) passes through unchanged.
    passthrough = _pca_aware_transform(x_pca, spca)
    assert np.allclose(passthrough, x_pca)


def test_apply_lognorm_false_skips_normalization():
    rng = np.random.default_rng(2)
    logn = rng.normal(size=(20, 8))  # already log-normalised features
    n_pcs = 3
    pca = PCA(n_components=n_pcs).fit(logn)
    pcs = pca.components_.T
    x_unwhit = (logn - pca.mean_) @ pcs
    xmean, xstd = x_unwhit.mean(0), x_unwhit.std(0)
    spca = SharedGenePCA(
        pcs=pcs,
        lognorm_mean=pca.mean_,
        xpca_mean=xmean,
        xpca_std=xstd,
        target_sum=1.0,
        apply_lognorm=False,
    )
    got = spca.transform(logn)
    assert np.allclose(got, (x_unwhit - xmean) / xstd, atol=1e-5)


def test_align_genes_reorders_to_fit_panel():
    spca = SharedGenePCA(
        pcs=np.eye(3),
        lognorm_mean=np.zeros(3),
        xpca_mean=np.zeros(3),
        xpca_std=np.ones(3),
        target_sum=1.0,
        var_names=["g0", "g1", "g2"],
    )
    genes_shuffled = np.array([[2.0, 0.0, 1.0]])  # in order g2, g0, g1
    aligned = spca.align_genes(genes_shuffled, ["g2", "g0", "g1"])
    assert np.allclose(aligned, [[0.0, 1.0, 2.0]])  # back to g0, g1, g2


def test_align_genes_raises_on_missing_gene():
    spca = SharedGenePCA(
        pcs=np.eye(2),
        lognorm_mean=np.zeros(2),
        xpca_mean=np.zeros(2),
        xpca_std=np.ones(2),
        target_sum=1.0,
        var_names=["g0", "g1"],
    )
    with pytest.raises(ValueError, match="missing fit genes"):
        spca.align_genes(np.zeros((1, 1)), ["g0"])


def test_coord_standardizer_roundtrip():
    cs = CoordStandardizer(mean=np.array([10.0, 20.0]), std=np.array([2.0, 4.0]))
    pos = np.array([[12.0, 24.0], [8.0, 16.0]])
    out = cs.transform(pos)
    assert np.allclose(out, [[1.0, 1.0], [-1.0, -1.0]])


class _FakeDS:
    """Minimal stand-in for H5ADDatasetDataclass with just the fields the
    builders read."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_shared_pca_from_dataclass_builds_and_errors():
    rng = np.random.default_rng(3)
    counts = rng.poisson(5, size=(40, 6)).astype(float)
    spca_ref, x_pca = _fit_recipe(counts, n_pcs=3)
    ds = _FakeDS(
        PCs=spca_ref.pcs,
        lognorm_mean=spca_ref.lognorm_mean,
        lognorm_target_sum=spca_ref.target_sum,
        var_names=["g%d" % i for i in range(6)],
        stats={"X_pca": {"mean": spca_ref.xpca_mean, "std": spca_ref.xpca_std}},
    )
    spca = shared_pca_from_dataclass(ds)
    assert np.allclose(spca.transform(counts), x_pca, atol=1e-4)

    # A pickle missing the recipe fields cannot back a SharedGenePCA.
    bad = _FakeDS(PCs=None, lognorm_mean=None, lognorm_target_sum=None, stats={})
    with pytest.raises(ValueError, match="cannot back a SharedGenePCA"):
        shared_pca_from_dataclass(bad)


def test_coord_standardizer_from_dataclass():
    ds = _FakeDS(
        timepoints_ordered=["A", "B"],
        stats={"coords": {"B": {"mean": np.array([1.0, 2.0]), "std": np.array([1.0, 1.0])}}},
    )
    cs = coord_standardizer_from_dataclass(ds)  # defaults to target tp "B"
    assert np.allclose(cs.transform([[2.0, 3.0]]), [[1.0, 1.0]])


def test_target_from_dataclass_shared_pca_projects_and_standardizes():
    """End-to-end (numpy): a shared-PCA target projects gene-space cells +
    standardises their coords."""
    from paired_slides_eval.contract import GeneratedSlide, TargetSlide
    from paired_slides_eval.reconcile import _standardize_generated_coords

    rng = np.random.default_rng(6)
    n_cells, n_genes, n_pcs = 25, 6, 3
    target_counts = rng.poisson(5, size=(n_cells, n_genes)).astype(float)
    spca, target_x_pca = _fit_recipe(target_counts, n_pcs)

    ds = _FakeDS(
        X_pca=target_x_pca,
        coords=rng.normal(size=(n_cells, 2)),
        ct=rng.integers(0, 3, size=n_cells),
        ct_ordered=["a", "b", "c"],
        ct_to_int={"a": 0, "b": 1, "c": 2},
        timepoints_ordered=["A", "B"],
        timepoint_indices={"B": np.arange(n_cells)},
        PCs=spca.pcs,
        lognorm_mean=spca.lognorm_mean,
        lognorm_target_sum=spca.target_sum,
        var_names=["g%d" % i for i in range(n_genes)],
        stats={
            "X_pca": {"mean": spca.xpca_mean, "std": spca.xpca_std},
            "coords": {"B": {"mean": np.array([1.0, 2.0]), "std": np.array([2.0, 4.0])}},
        },
    )

    target = TargetSlide.from_dataclass(ds, shared_pca=True)
    assert isinstance(target.pca, SharedGenePCA)
    assert target.n_classes == 3

    # gene-space generated cells project through the SHARED basis (match SharedGenePCA directly)
    gen_counts = rng.poisson(5, size=(10, n_genes)).astype(float)
    raw_pos = rng.normal(size=(10, 2)) * 5 + 3
    gen = GeneratedSlide(x=gen_counts, pos=raw_pos).project(target.pca)
    assert gen.x.shape == (10, n_pcs)
    assert np.allclose(gen.x, spca.transform(gen_counts), atol=1e-4)

    # ...and raw coords get standardised into the target frame
    gen = _standardize_generated_coords(gen, target.coord_transform)
    assert np.allclose(gen.pos, (raw_pos - [1.0, 2.0]) / [2.0, 4.0], atol=1e-5)


def test_auto_coord_detection_and_reconcile():
    """Coords='auto' standardises raw generated coords and passes standardised
    ones through."""
    from paired_slides_eval.contract import GeneratedSlide, TargetSlide
    from paired_slides_eval.reconcile import _detect_coord_space, _reconcile_generated

    ct = CoordStandardizer(mean=np.array([100.0, 200.0]), std=np.array([10.0, 20.0]))
    rng = np.random.default_rng(0)
    raw = rng.normal([100.0, 200.0], [10.0, 20.0], size=(500, 2))  # ~ target raw frame
    standardized = rng.normal(0.0, 1.0, size=(500, 2))  # already ~ unit

    assert _detect_coord_space(raw, ct) == "standardize"
    assert _detect_coord_space(standardized, ct) == "passthrough"

    target = TargetSlide(x=np.zeros((3, 4)), pos=np.zeros((3, 2)), coord_transform=ct)

    out, notes = _reconcile_generated(
        GeneratedSlide(x=np.zeros((500, 4)), pos=raw),
        target,
        coords="auto",
    )
    assert np.allclose(out.pos.std(axis=0), 1.0, atol=0.3)  # mapped into the standardised frame
    assert notes and "standardize" in notes[0]

    out2, _ = _reconcile_generated(
        GeneratedSlide(x=np.zeros((500, 4)), pos=standardized),
        target,
        coords="auto",
    )
    assert np.allclose(out2.pos, standardized)  # already standardised -> passthrough

    # No coord frame on the target -> nothing to reconcile (legacy h5ad path)
    bare = TargetSlide(x=np.zeros((3, 4)), pos=np.zeros((3, 2)))
    out3, notes3 = _reconcile_generated(
        GeneratedSlide(x=np.zeros((5, 4)), pos=np.zeros((5, 2))),
        bare,
        coords="auto",
    )
    assert notes3 == []


def test_from_dataclass_shared_pca_auto_falls_back_without_recipe():
    """shared_pca='auto' attaches the transform iff the pickle carries the
    recipe (else pca=None)."""
    from paired_slides_eval.contract import TargetSlide

    bare = _FakeDS(
        X_pca=np.zeros((4, 3)),
        coords=np.zeros((4, 2)),
        ct=np.array([0, 1, 0, 1]),
        ct_ordered=["a", "b"],
        ct_to_int={"a": 0, "b": 1},
        timepoints_ordered=["A", "B"],
        timepoint_indices={"B": np.arange(4)},
        # no PCs / lognorm_* / stats recipe
        PCs=None,
        lognorm_mean=None,
        lognorm_target_sum=None,
        var_names=None,
        stats={},
    )
    target = TargetSlide.from_dataclass(bare, shared_pca="auto")
    assert target.pca is None and target.coord_transform is None  # graceful fallback, no raise


def test_fixed_reference_accuracy_is_model_independent():
    torch = pytest.importorskip("torch")
    from paired_slides_eval.metrics.classifier_gap import fixed_reference_accuracy

    rng = np.random.default_rng(5)
    n, n_pcs, n_classes = 200, 4, 3
    real_x = rng.normal(size=(n, n_pcs)).astype(np.float32)
    real_pos = rng.normal(size=(n, 2)).astype(np.float32)
    real_ct = rng.integers(0, n_classes, size=n)

    # Gene-only (spatial=False) classifier: reads the centroid's n_pcs features.
    clf = torch.nn.Linear(n_pcs, n_classes)
    clf.output_dim = n_classes

    acc1 = fixed_reference_accuracy(
        real_x,
        real_pos,
        real_ct,
        clf,
        spatial=False,
        n_centroids=50,
        seed=7,
    )
    acc2 = fixed_reference_accuracy(
        real_x,
        real_pos,
        real_ct,
        clf,
        spatial=False,
        n_centroids=50,
        seed=7,
    )
    assert acc1 == acc2  # deterministic given (classifier, target, seed) — model-independent
    assert 0.0 <= acc1 <= 1.0
