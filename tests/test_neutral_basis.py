"""Neutral-basis P* — the fair, model-neutral scoring space (docs/neutral_basis_eval_plan.md).

Pure numpy/sklearn (no torch): the P* builder, k slicing, and the symmetric-projection reorder.
"""

from __future__ import annotations

import numpy as np
import pytest

from paired_slides_eval.contract import TargetSlide, _pca_aware_transform
from paired_slides_eval.data.dataclass import slide_expression_matrix
from paired_slides_eval.data.neutral_basis import (
    fit_neutral_basis,
    neutral_basis_from_dataclass,
    pca_knee,
)
from paired_slides_eval.data.shared_pca import GenPCAInversion

G, K = 12, 6


class _DS:
    """Minimal H5ADDatasetDataclass stand-in carrying just the neutral-basis fields."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _ds_with_neutral(seed=0):
    rng = np.random.default_rng(seed)
    Lt = rng.normal(size=(40, G))  # target log-gene
    Ls = rng.normal(size=(25, G))  # source log-gene
    mean = Lt.mean(0)
    _, _, Vt = np.linalg.svd(Lt - mean, full_matrices=False)
    pcs = Vt[:K].T  # (G, K)
    L = np.concatenate([Ls, Lt], 0)
    neutral_x = ((L - mean) @ pcs).astype(np.float32)
    return _DS(
        neutral_pcs=pcs, neutral_mean=mean, neutral_target_sum=1e4,
        neutral_x=neutral_x, neutral_k=4, var_names=[f"g{i}" for i in range(G)],
        X_pca=rng.normal(size=(65, 50)),
        timepoints_ordered=["A", "B"],
        timepoint_indices={"A": np.arange(25), "B": np.arange(25, 65)},
        coords=rng.normal(size=(65, 2)), ct=np.array(["x"] * 65),
        ct_to_int={"x": 0}, ct_ordered=["x"],
        stats={"coords": {"B": {"mean": np.zeros(2), "std": np.ones(2)}}, "X_pca": {}},
    ), Lt


def test_pca_knee_bounds():
    ev = np.array([50.0, 20.0, 8.0, 3.0, 1.0] + [0.1] * 45)
    assert 5 <= pca_knee(ev) <= 50


def test_fit_neutral_basis_extracted():
    """The model-neutral fit: fit on target rows, project all cells, pick the knee."""
    rng = np.random.default_rng(3)
    log_gene = rng.normal(size=(60, G))
    target_idx = np.arange(20, 60)
    basis, scores = fit_neutral_basis(
        log_gene, target_idx, n_pcs=G, target_sum=1e4, var_names=[f"g{i}" for i in range(G)]
    )
    assert basis.pcs.shape[0] == G and basis.target_sum == 1e4 and 5 <= basis.k <= G
    assert np.allclose(scores, basis.project(log_gene), atol=1e-6)  # scores == all-cell projection
    assert np.allclose(scores[20:], basis.project(log_gene[20:]), atol=1e-6)  # target rows


def test_neutral_basis_reproduces_stored_scores():
    ds, Lt = _ds_with_neutral()
    pstar = neutral_basis_from_dataclass(ds, k=K)
    assert pstar.components.shape == (K, G)  # out_dim=K, in_dim=G, unwhitened
    assert np.allclose(pstar.transform_lognorm(Lt), ds.neutral_x[25:], atol=1e-5)


def test_slice_to_headline_k():
    ds, _ = _ds_with_neutral()
    assert neutral_basis_from_dataclass(ds).components.shape == (4, G)  # defaults to neutral_k
    assert slide_expression_matrix(ds).shape == (65, 4)
    assert slide_expression_matrix(ds, k=6).shape == (65, 6)


def test_slide_expression_falls_back_to_xpca():
    ds = _DS(X_pca=np.zeros((10, 5)), neutral_x=None)
    assert slide_expression_matrix(ds).shape == (10, 5)


def test_source_pca_beats_passthrough_after_reorder():
    """A model carrying its own inverse inverts even when its width == the neutral out_dim."""
    ds, _ = _ds_with_neutral()
    pstar = neutral_basis_from_dataclass(ds, k=K)  # out_dim == K
    rng = np.random.default_rng(1)
    inv = GenPCAInversion(
        components=rng.normal(size=(K, G)), mean=np.zeros(G),
        sc_mean=np.zeros(K), sc_scale=np.ones(K), var_names=ds.var_names, target_sum=1e4,
    )
    z = rng.normal(size=(5, K))  # width == out_dim AND == inv.n_pcs
    out = _pca_aware_transform(z, pstar, inv)
    assert not np.allclose(out, z)  # inverted+reprojected, NOT passed through
    manual = pstar.transform_lognorm(pstar.align_genes(inv.to_log_gene(z), inv.var_names))
    assert np.allclose(out, manual, atol=1e-6)


def test_target_from_dataclass_in_neutral_space():
    ds, _ = _ds_with_neutral()
    t = TargetSlide.from_dataclass(ds, k=4)
    assert t.x.shape == (40, 4)  # target slide (B) at k=4
    assert t.pca.components.shape == (4, G)


def test_missing_neutral_basis_raises():
    with pytest.raises(ValueError, match="no neutral basis"):
        neutral_basis_from_dataclass(_DS(neutral_pcs=None))
