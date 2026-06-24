"""Test the classifier accuracy-gap metric with a tiny torch classifier (skips without torch)."""

import pytest

torch = pytest.importorskip("torch")

from paired_slides_eval.metrics.classifier_gap import classifier_accuracy_gap


class _TinyClassifier(torch.nn.Module):
    """A trivial spatial classifier: pools points, linear head. output_dim cell types."""

    def __init__(self, point_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.lin = torch.nn.Linear(point_dim, output_dim)

    def forward(self, x):  # x: (B, k, point_dim)
        return self.lin(x.mean(dim=1))


def test_accuracy_gap_keys_and_bounds(rng):
    b, n, n_pcs = 30, 6, 5
    gen_x = rng.normal(size=(b, n, n_pcs)).astype("float32")
    gen_pos = rng.uniform(0, 5, size=(b, n, 2)).astype("float32")
    gt_x = rng.normal(size=(b, n, n_pcs)).astype("float32")
    gt_pos = rng.uniform(0, 5, size=(b, n, 2)).astype("float32")
    gt_ct = rng.integers(0, 4, size=b)

    clf = _TinyClassifier(point_dim=n_pcs + 2, output_dim=4)
    clf.n_neighbors = n  # so _resolve_n_neighbors doesn't warn/fallback

    out = classifier_accuracy_gap(
        gen_x, gen_pos, gt_x, gt_pos, gt_ct, clf, prefix="test", spatial=True
    )
    assert set(out) == {"test/ct/acc_real", "test/ct/acc_gen", "test/ct/acc_gap"}
    assert out["test/ct/acc_gap"] == pytest.approx(
        abs(out["test/ct/acc_real"] - out["test/ct/acc_gen"])
    )
    for v in out.values():
        assert 0.0 <= v <= 1.0


def test_accuracy_gap_identical_inputs_zero_gap(rng):
    b, n, n_pcs = 20, 5, 4
    x = rng.normal(size=(b, n, n_pcs)).astype("float32")
    pos = rng.uniform(0, 5, size=(b, n, 2)).astype("float32")
    gt_ct = rng.integers(0, 3, size=b)
    clf = _TinyClassifier(point_dim=n_pcs + 2, output_dim=3)
    clf.n_neighbors = n
    # gen == gt -> identical predictions -> zero gap
    out = classifier_accuracy_gap(x, pos, x.copy(), pos.copy(), gt_ct, clf, spatial=True)
    assert out["ct/acc_gap"] == pytest.approx(0.0)
