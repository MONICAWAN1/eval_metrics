"""Masked-centroid expression reconstruction metrics."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from paired_slides_eval import GeneratedSlide, TargetSlide, evaluate
from paired_slides_eval.metrics.expr_recon import expr_recon_gap, fixed_reference_mse


class _MeanNeighborRegressor(torch.nn.Module):
    """Tiny regressor that predicts the centroid from the mean of its neighbours."""

    def __init__(self, n_pcs: int, n_neighbors: int = 2):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.output_dim = n_pcs
        self.bias = torch.nn.Parameter(torch.zeros(n_pcs))

    def forward(self, x):
        return x[:, 1:, :].mean(dim=1) + self.bias


def test_expr_recon_gap_zero_on_identical_niches():
    reg = _MeanNeighborRegressor(n_pcs=3, n_neighbors=2)
    x = np.zeros((5, 3, 3), dtype=np.float32)
    pos = np.zeros((5, 3, 2), dtype=np.float32)
    pos[:, :, 0] = np.array([0.0, 1.0, 2.0])

    out = expr_recon_gap(x, pos, x.copy(), pos.copy(), reg, prefix="test")

    assert out["test/recon/mse_real"] == 0.0
    assert out["test/recon/mse_gen"] == 0.0
    assert out["test/recon/mse_gap"] == 0.0


def test_expr_recon_gap_increases_for_shifted_generated_centres():
    reg = _MeanNeighborRegressor(n_pcs=3, n_neighbors=2)
    real_x = np.zeros((5, 3, 3), dtype=np.float32)
    gen_x = real_x.copy()
    gen_x[:, 0, :] = 1.0
    pos = np.zeros((5, 3, 2), dtype=np.float32)
    pos[:, :, 0] = np.array([0.0, 1.0, 2.0])

    out = expr_recon_gap(gen_x, pos, real_x, pos.copy(), reg, prefix="test")

    assert out["test/recon/mse_real"] == 0.0
    assert out["test/recon/mse_gen"] == pytest.approx(1.0)
    assert out["test/recon/mse_gap"] == pytest.approx(1.0)


def test_fixed_reference_mse_uses_seeded_real_niches():
    reg = _MeanNeighborRegressor(n_pcs=2, n_neighbors=2)
    real_pos = np.stack([np.arange(12), np.zeros(12)], axis=1).astype(np.float32)
    real_x = np.zeros((12, 2), dtype=np.float32)

    assert fixed_reference_mse(real_x, real_pos, reg, n_centroids=5, seed=0) == 0.0


def test_evaluate_recon_auto_builds_flat_niches_with_fixed_real_ref():
    reg = _MeanNeighborRegressor(n_pcs=2, n_neighbors=2)
    target_pos = np.stack([np.arange(8), np.zeros(8)], axis=1).astype(np.float32)
    target = TargetSlide(x=np.zeros((8, 2), dtype=np.float32), pos=target_pos)
    generated = GeneratedSlide(
        x=np.ones((8, 2), dtype=np.float32),
        pos=target_pos.copy(),
    )

    out = evaluate(target, generated, regressor=reg, groups=("recon",), recon_real_n=4, seed=0)

    assert out["test/recon/mse_real"] == 0.0
    assert out["test/recon/mse_gen"] == pytest.approx(1.0)
    assert out["test/recon/mse_gap"] == pytest.approx(1.0)
    assert not out["_skipped"]
    assert any("recon/mse_real from a fixed seeded sample" in note for note in out["_notes"])
