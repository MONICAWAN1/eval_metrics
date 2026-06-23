import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nicheflow_eval.metrics.concordance import cell_type_concordance


class _GeneOnlyNet(torch.nn.Module):
    """Minimal gene-only classifier (expression -> logits) for testing concordance."""

    def __init__(self, n_pcs: int, n_classes: int):
        super().__init__()
        self.output_dim = n_classes
        self.fc = torch.nn.Linear(n_pcs, n_classes)

    def forward(self, x):
        return self.fc(x)


def test_concordance_keys_and_ranges(real_slide, generated_niches):
    # The neutral classifier labels both the generated niche and its paired real niche (gt_*).
    rng = np.random.default_rng(0)
    gen_x, gen_pos = generated_niches["x"], generated_niches["pos"]
    gt_x = gen_x + rng.normal(scale=0.1, size=gen_x.shape)      # stand-in paired real niche
    gt_pos = gen_pos + rng.normal(scale=0.1, size=gen_pos.shape)

    n_pcs = gen_x.shape[-1]
    clf = _GeneOnlyNet(n_pcs, real_slide["n_classes"])
    out = cell_type_concordance(
        gen_x, gen_pos, gt_x, gt_pos, clf,
        prefix="test", spatial=False, n_classes=real_slide["n_classes"],
    )
    assert set(out) == {
        "test/ct/f1", "test/ct/acc", "test/ct/prop_kl", "test/ct/prop_tv", "test/ct/prop_jsd"
    }
    assert 0.0 <= out["test/ct/acc"] <= 1.0
    assert 0.0 <= out["test/ct/prop_tv"] <= 1.0
    assert out["test/ct/prop_kl"] >= 0.0
    assert out["test/ct/prop_jsd"] >= 0.0


def test_concordance_perfect_on_identical_niches(real_slide, generated_niches):
    # Identical generated & real niches -> identical labels -> perfect agreement, zero divergence.
    gen_x, gen_pos = generated_niches["x"], generated_niches["pos"]
    clf = _GeneOnlyNet(gen_x.shape[-1], real_slide["n_classes"])
    out = cell_type_concordance(
        gen_x, gen_pos, gen_x.copy(), gen_pos.copy(), clf,
        prefix="test", spatial=False, n_classes=real_slide["n_classes"],
    )
    assert out["test/ct/acc"] == 1.0
    assert out["test/ct/prop_tv"] == 0.0
