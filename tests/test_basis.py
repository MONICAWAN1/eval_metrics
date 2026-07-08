"""The shared-space `Basis` — one definition of expression + coordinate
transforms.

Pure numpy: `from_dataclass`, `apply`, and `to_fm_npz` byte-identity to the legacy mapping.

"""

import numpy as np

from paired_slides_eval.data.shared_pca import (
    Basis,
    coord_standardizer_from_dataclass,
    shared_pca_from_dataclass,
)


class _DS:
    """Minimal stand-in carrying the recipe fields `from_dataclass` reads."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _ds(G=6, k=3):
    rng = np.random.default_rng(0)
    return _DS(
        PCs=rng.normal(size=(G, k)),
        lognorm_mean=rng.normal(size=G),
        lognorm_target_sum=1e4,
        var_names=[f"g{i}" for i in range(G)],
        timepoints_ordered=["A", "B"],
        stats={
            "X_pca": {"mean": rng.normal(size=k), "std": rng.random(k) + 0.5},
            "coords": {"B": {"mean": np.array([10.0, 20.0]), "std": np.array([2.0, 4.0])}},
        },
    )


def test_from_dataclass_matches_components():
    ds = _ds()
    b = Basis.from_dataclass(ds)
    sp = shared_pca_from_dataclass(ds)
    cs = coord_standardizer_from_dataclass(ds, "B")
    assert np.array_equal(b.expression.pcs, sp.pcs) and b.expression.target_sum == sp.target_sum
    assert np.array_equal(b.coords.mean, cs.mean) and np.array_equal(b.coords.std, cs.std)


def test_apply_matches_component_transforms():
    b = Basis.from_dataclass(_ds())
    rng = np.random.default_rng(1)
    genes = rng.random((5, 6)) * 10
    coords = rng.normal(size=(5, 2))
    x, c = b.apply(genes, coords)
    assert np.allclose(x, b.expression.transform(genes))
    assert np.allclose(c, b.coords.transform(coords))
    assert b.apply(coords=coords)[0] is None  # partial application
    assert b.apply(genes=genes)[1] is None


def test_to_fm_npz_byte_identical_to_legacy_mapping(tmp_path):
    b = Basis.from_dataclass(_ds())
    sp, cs = b.expression, b.coords
    out = str(tmp_path / "shared.npz")
    b.to_fm_npz(out)

    # The exact mapping otcfm_export used to build by hand — must match key-for-key.
    expected = {
        "space": "pca",
        "n_pcs": int(np.asarray(sp.pcs).shape[1]),
        "whiten": True,
        "pca_components": np.asarray(sp.pcs, dtype=np.float32).T,
        "pca_mean": np.asarray(sp.lognorm_mean, dtype=np.float32),
        "sc_mean": np.asarray(sp.xpca_mean, dtype=np.float32),
        "sc_scale": np.asarray(sp.xpca_std, dtype=np.float32),
        "target_sum": np.float32(sp.target_sum),
        "var_names": np.asarray([str(v) for v in sp.var_names]),
        "coord_mean": np.asarray(cs.mean, dtype=np.float32).ravel(),
        "coord_std": np.asarray(cs.std, dtype=np.float32).ravel(),
    }
    loaded = np.load(out, allow_pickle=True)
    assert set(loaded.files) == set(expected)
    for key, ev in expected.items():
        got = loaded[key]
        if isinstance(ev, np.ndarray):
            assert np.array_equal(got, ev), key
        else:
            assert got.item() == ev, key
