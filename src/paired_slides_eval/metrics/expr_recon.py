"""Gene-expression reconstruction gap (``recon/mse_real``, ``recon/mse_gen``, ``recon/mse_gap``).

A masked-centroid expression regressor (trained on a held-out/target slide to predict a cell's
expression from its spatial KNN neighbours) is run on the real target niches and on the generated
niches; the statistic is the reconstruction MSE on each. If the generated local spatial-expression
structure is realistic, the real-trained regressor predicts generated centres about as well as real
ones, so a small ``|mse_real - mse_gen|`` gap is good. This is the regression analog of
:mod:`paired_slides_eval.metrics.classifier_gap`; the niche assembly is shared.
"""

from __future__ import annotations

import numpy as np

from paired_slides_eval.metrics._common import (
    build_microenv_points,
    build_paired_niches_from_flat,
)
from paired_slides_eval.metrics.concordance import _resolve_n_neighbors


def _predict_center(regressor, feats: np.ndarray) -> np.ndarray:
    """Run the frozen regressor over masked-centroid neighbour sets -> predicted centre expression."""
    import torch

    regressor.eval()
    device = next(regressor.parameters()).device
    with torch.no_grad():
        return (
            regressor(torch.as_tensor(feats, dtype=torch.float32, device=device)).cpu().numpy()
        )


def expr_recon_gap(
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    gt_x: np.ndarray,
    gt_pos: np.ndarray,
    regressor,
    *,
    prefix: str = "",
    n_neighbors: int | None = None,
) -> dict[str, float]:
    """Reconstruction MSE of ``regressor`` on real vs. generated niches, and the gap.

    The regressor predicts each centroid's expression from its (masked) KNN neighbours; the target is
    the centroid's own expression (point 0 of the niche). ``n_neighbors`` is the KNN ``k`` — ``None``
    uses the value recorded on the regressor at training time.

    Returns ``{prefix/recon/mse_real, prefix/recon/mse_gen, prefix/recon/mse_gap}``.
    """
    p = f"{prefix}/" if prefix else ""
    k = _resolve_n_neighbors(n_neighbors, regressor)

    feats_gen = build_microenv_points(gen_x, gen_pos, k)
    feats_real = build_microenv_points(gt_x, gt_pos, k)
    center_gen = np.asarray(gen_x)[:, 0, :]
    center_real = np.asarray(gt_x)[:, 0, :]

    mse_gen = float(np.mean((_predict_center(regressor, feats_gen) - center_gen) ** 2))
    mse_real = float(np.mean((_predict_center(regressor, feats_real) - center_real) ** 2))
    return {
        f"{p}recon/mse_real": mse_real,
        f"{p}recon/mse_gen": mse_gen,
        f"{p}recon/mse_gap": abs(mse_gen - mse_real),
    }


def fixed_reference_mse(
    real_x: np.ndarray,
    real_pos: np.ndarray,
    regressor,
    *,
    n_neighbors: int | None = None,
    n_centroids: int = 2000,
    seed: int = 0,
) -> float:
    """Reconstruction MSE on a **model-independent** seeded sample of real target niches.

    The paired ``recon/mse_real`` is measured on whichever real niches got paired to a model's
    generated centroids, so it drifts between models. This instead samples a fixed, seeded set of
    target cells as centroids and reconstructs their expression from real neighbours — making
    ``mse_real`` a constant property of (regressor, target, niche size), comparable across every
    model. Used by :func:`~paired_slides_eval.evaluate.evaluate` when ``recon_real_reference='fixed'``.
    """
    real_pos = np.asarray(real_pos)
    n_real = len(real_pos)
    rng = np.random.default_rng(seed)
    centroids = (
        rng.choice(n_real, n_centroids, replace=False) if n_real > n_centroids else np.arange(n_real)
    )

    k = _resolve_n_neighbors(n_neighbors, regressor)
    niche_x, niche_pos, _, _, _ = build_paired_niches_from_flat(
        real_x, real_pos, real_x, real_pos, k, centroid_indices=centroids
    )
    feats = build_microenv_points(niche_x, niche_pos, k)
    center = niche_x[:, 0, :]
    return float(np.mean((_predict_center(regressor, feats) - center) ** 2))
