"""Flat whole-slide -> niche reconstruction for the classifier metrics.

A whole-slide model emits a flat `GeneratedSlide` (no niche structure). The classifier groups
(`concordance`, `ct_gap`) need paired microenvironments; these tests cover both the pure-numpy
geometry builder and `evaluate` auto-building niches from a flat slide.

"""

import numpy as np
import pytest

from paired_slides_eval import GeneratedSlide, TargetSlide, evaluate
from paired_slides_eval.metrics._common import (
    build_paired_niches_from_flat,
    build_paired_niches_from_flat_fixed_centroids,
)


def test_build_paired_niches_shapes_and_centroids(rng):
    gen_pos = rng.uniform(0, 10, size=(50, 2))
    gen_x = rng.normal(size=(50, 4))
    real_pos = rng.uniform(0, 10, size=(80, 2))
    real_x = rng.normal(size=(80, 4))
    real_ct = rng.integers(0, 3, size=80)
    k = 6

    nx, npos, gtx, gtpos, gtct = build_paired_niches_from_flat(
        gen_x,
        gen_pos,
        real_x,
        real_pos,
        k,
        real_ct=real_ct,
    )
    # k = number of NEIGHBOURS, so each niche has k+1 points (centroid + k).
    assert nx.shape == (50, k + 1, 4) and npos.shape == (50, k + 1, 2)
    assert gtx.shape == (50, k + 1, 4) and gtpos.shape == (50, k + 1, 2)
    assert gtct.shape == (50,)

    # Generated niche point 0 is the centroid generated cell itself.
    assert np.allclose(npos[:, 0], gen_pos)
    assert np.allclose(nx[:, 0], gen_x)
    # Within a niche the points are distance-sorted from the centroid.
    d = np.linalg.norm(npos - npos[:, :1], axis=-1)
    assert np.all(np.diff(d, axis=1) >= -1e-9)
    # Real niche centroid + label come from the nearest real cell to the generated centroid.
    for b in range(5):
        r0 = np.linalg.norm(real_pos - gen_pos[b], axis=1).argmin()
        assert np.allclose(gtpos[b, 0], real_pos[r0])
        assert gtct[b] == real_ct[r0]


def test_build_paired_niches_clamps_k_and_subsamples(rng):
    gen_x, gen_pos = rng.normal(size=(3, 4)), rng.uniform(size=(3, 2))
    real_x, real_pos = rng.normal(size=(2, 4)), rng.uniform(size=(2, 2))
    # k clamps PER SIDE to the available neighbours (n-1): gen 3 cells -> 3 points, real 2 -> 2.
    nx, npos, gtx, *_ = build_paired_niches_from_flat(gen_x, gen_pos, real_x, real_pos, 6)
    assert nx.shape == (3, 3, 4) and gtx.shape == (3, 2, 4)
    # centroid_indices restricts which generated cells become niches (k=2 -> 3 points).
    nx2, *_, gtct = build_paired_niches_from_flat(
        rng.normal(size=(20, 4)),
        rng.uniform(size=(20, 2)),
        real_x,
        real_pos,
        2,
        centroid_indices=np.arange(5),
    )
    assert nx2.shape == (5, 3, 4) and gtct is None


def test_fixed_target_centroid_ot_pairing_keeps_target_label_order():
    real_pos = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    real_x = np.arange(12, dtype=float).reshape(3, 4)
    real_ct = np.array([1, 2, 3])
    # Generated cells are close to fixed target centroids 1 and 0, in that order.
    gen_pos = np.array([[9.5, 0.0], [0.5, 0.0], [30.0, 0.0]])
    gen_x = np.arange(100, 112, dtype=float).reshape(3, 4)

    nx, npos, gtx, gtpos, gtct = build_paired_niches_from_flat_fixed_centroids(
        gen_x,
        gen_pos,
        real_x,
        real_pos,
        k=1,
        real_ct=real_ct,
        target_centroid_indices=np.array([0, 1]),
    )

    # Rows stay anchored to target centroid order [0, 1], while generated centroids are OT matched
    # uniquely to the nearby generated cells [1, 0].
    assert np.allclose(npos[:, 0], gen_pos[[1, 0]])
    assert np.allclose(gtpos[:, 0], real_pos[[0, 1]])
    assert np.array_equal(gtct, np.array([1, 2]))
    assert nx.shape == (2, 2, 4) and gtx.shape == (2, 2, 4)


pytest.importorskip("torch")  # the evaluate-level tests below need a tiny torch classifier
import torch  # noqa: E402


class _TinyClassifier(torch.nn.Module):
    def __init__(self, point_dim, output_dim, n_neighbors):
        super().__init__()
        self.output_dim = output_dim
        self.n_neighbors = n_neighbors
        self.lin = torch.nn.Linear(point_dim, output_dim)

    def forward(self, x):  # x: (B, k, point_dim)
        return self.lin(x.mean(dim=1))


def _flat_target_and_slide(real_slide):
    target = TargetSlide(
        x=real_slide["x"],
        pos=real_slide["pos"],
        ct=real_slide["ct"],
        n_classes=real_slide["n_classes"],
    )
    rng = np.random.default_rng(1)
    n_pcs = real_slide["x"].shape[1]
    pos = rng.uniform(0, 10, size=(120, 2))
    x = rng.normal(size=(120, n_pcs)) + 0.3 * pos[:, :1]
    return target, GeneratedSlide(x=x, pos=pos)


def test_evaluate_auto_builds_niches_from_flat_slide(real_slide):
    target, generated = _flat_target_and_slide(real_slide)
    k = 8
    clf = _TinyClassifier(real_slide["x"].shape[1], real_slide["n_classes"], n_neighbors=k)

    out = evaluate(target, generated, classifier=clf, groups=("concordance", "ct_gap"))
    # Both classifier groups now run on the flat slide (no GeneratedNiches supplied).
    assert "test/ct/acc" in out and "test/ct/acc_gap" in out
    assert not out["_skipped"]
    assert out["_notes"] and "auto-built" in out["_notes"][0]


def test_evaluate_uses_fixed_target_centroid_ot_for_flat_slide(real_slide):
    target, generated = _flat_target_and_slide(real_slide)
    target.eval_centroid_indices = np.arange(10)
    k = 8
    clf = _TinyClassifier(real_slide["x"].shape[1], real_slide["n_classes"], n_neighbors=k)

    out = evaluate(target, generated, classifier=clf, groups=("concordance", "ct_gap"))

    assert "test/ct/acc" in out and "test/ct/acc_gap" in out
    assert any("fixed target centroids + OT" in note for note in out["_notes"])


def test_evaluate_auto_niche_can_be_disabled(real_slide):
    target, generated = _flat_target_and_slide(real_slide)
    clf = _TinyClassifier(real_slide["x"].shape[1], real_slide["n_classes"], n_neighbors=8)
    out = evaluate(
        target,
        generated,
        classifier=clf,
        groups=("concordance", "ct_gap"),
        auto_niche_from_flat=False,
    )
    # With auto-build off, a flat slide has no paired niches -> both skipped.
    assert "test/ct/acc" not in out
    assert len(out["_skipped"]) == 2


def test_evaluate_flat_ct_gap_needs_target_labels(real_slide):
    target, generated = _flat_target_and_slide(real_slide)
    target.ct = None  # no labels -> ct_gap cannot score, concordance still can
    target.n_classes = None
    clf = _TinyClassifier(real_slide["x"].shape[1], 4, n_neighbors=6)
    out = evaluate(target, generated, classifier=clf, groups=("concordance", "ct_gap"))
    assert "test/ct/acc" in out  # concordance ran
    assert any("ct_gap" in s for s in out["_skipped"])  # ct_gap skipped (no gt_ct)


def test_evaluate_regression_stays_skipped_for_flat(real_slide):
    target, generated = _flat_target_and_slide(real_slide)
    out = evaluate(target, generated, groups=("regression",))
    assert any("regression" in s for s in out["_skipped"])
