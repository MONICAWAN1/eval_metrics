"""Smoke test the NicheFlow adapter preprocessing (needs torch + scanpy; skips otherwise)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("scanpy")
ad = pytest.importorskip("anndata")

import pandas as pd

from paired_slides_eval.adapters.nicheflow import preprocess_pair, target_from_dataclass


def _toy_slide(rng, n=120, n_genes=30):
    counts = rng.poisson(2.0, size=(n, n_genes)).astype("float32")
    a = ad.AnnData(X=counts)
    a.obsm["spatial"] = rng.uniform(0, 5, size=(n, 2)).astype("float32")
    a.obs["class"] = pd.Categorical(rng.choice(list("abc"), size=n))
    a.var_names = [f"g{i}" for i in range(n_genes)]
    return a


def test_preprocess_pair_builds_dataclass(tmp_path, rng):
    a = _toy_slide(rng)
    b = _toy_slide(rng)
    pa, pb = tmp_path / "a.h5ad", tmp_path / "b.h5ad"
    a.write_h5ad(pa)
    b.write_h5ad(pb)

    ds, pre = preprocess_pair(pa, pb, n_pcs=10, radius=1.0, dx=1.0, dy=1.0)
    assert ds.timepoints_ordered == ["A", "B"]
    assert ds.X_pca.shape[1] == 10
    assert set(ds.timepoint_indices) == {"A", "B"}
    # radius graph + grid subsample populated for both slides
    for t in ("A", "B"):
        assert t in ds.timepoint_neighboring_indices
        assert len(ds.subsampled_timepoint_idx[t]) > 0
    assert ds.test_microenvs == max(len(v) for v in ds.subsampled_timepoint_idx.values())

    # target_from_dataclass reads the standardized X_pca space.
    target = target_from_dataclass(ds, timepoint="B")
    assert target.x.shape[1] == 10
    assert target.n_classes == len(ds.ct_to_int)
