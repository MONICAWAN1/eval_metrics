"""Nearest-neighbour two-sample test, pooled over spatial niches (``c2st/nn*``).

A forgiving, label-free spatial check. Real and generated cells are pooled (balanced to equal counts)
in one expression space; each cell's nearest neighbour in that pool tells whether it "looks real"
(its NN is a real cell) or "looks generated". For each generated cell — a niche centroid — we average
that bit over its nearest *spatial* neighbours, giving a per-niche real-fraction ``p_real``. The slide
score is the mean over niches:

* ``p_real ~ 0.5``  -> generated neighbourhoods are indistinguishable from real (good);
* ``p_real -> 0``   -> generated cells neighbour other generated cells (separable);
* ``p_real -> 1``   -> generated cells sit on real cells (memorisation).

A continuous statistic (not a hard majority vote) is reported with its spread, so a mean near 0.5
from genuine mixing is distinguishable from one produced by a bimodal, regionally-separable slide.

"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from paired_slides_eval.metrics._common import knn_indices


def c2st_nn(
    real_x: np.ndarray,
    real_pos: np.ndarray,
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    *,
    spatial_k: int = 10,
    max_n: int = 2000,
    z_score: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-niche real-fractions ``(p_real_gen, p_real_real)`` for
    generated and real centroids.

    Real and generated are subsampled to equal counts so the chance level is 0.5. For every cell, the
    nearest neighbour in the joint expression pool (excluding itself) is looked up; a niche's value is
    the fraction of its ``spatial_k`` nearest spatial neighbours whose such NN is a real cell.

    """
    rng = np.random.default_rng(seed)
    real_x = np.asarray(real_x, dtype=np.float64)
    gen_x = np.asarray(gen_x, dtype=np.float64)
    real_pos = np.asarray(real_pos, dtype=np.float64)
    gen_pos = np.asarray(gen_pos, dtype=np.float64)

    m = int(min(len(real_x), len(gen_x), max_n))
    ri = rng.choice(len(real_x), m, replace=False)
    gi = rng.choice(len(gen_x), m, replace=False)
    rx, rp = real_x[ri], real_pos[ri]
    gx, gp = gen_x[gi], gen_pos[gi]

    if z_score:
        mean = rx.mean(axis=0)
        std = rx.std(axis=0) + 1e-8
        rx, gx = (rx - mean) / std, (gx - mean) / std

    # Joint pool: rows [0, m) are real (label 0), rows [m, 2m) are generated (label 1).
    tree = cKDTree(np.concatenate([rx, gx], axis=0))

    def nn_is_real(query: np.ndarray, offset: int) -> np.ndarray:
        # nearest neighbour of each row in the pool, excluding itself by *index* (ties at distance 0,
        # e.g. memorised copies, can otherwise put self in either column).
        _, idx = tree.query(query, k=2)
        self_idx = offset + np.arange(len(query))
        nn = np.where(idx[:, 0] == self_idx, idx[:, 1], idx[:, 0])
        return nn < m

    real_bit = nn_is_real(rx, 0)
    gen_bit = nn_is_real(gx, m)

    def per_niche(pos: np.ndarray, bit: np.ndarray) -> np.ndarray:
        nbr = knn_indices(pos, spatial_k)[:, 1:]  # spatial neighbours per centroid (drop self)
        return bit[nbr].mean(axis=1)

    return per_niche(gp, gen_bit), per_niche(rp, real_bit)


def c2st_nn_metrics(
    real_x: np.ndarray,
    real_pos: np.ndarray,
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    *,
    prefix: str = "",
    spatial_k: int = 10,
    max_n: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Spatially-pooled nearest-neighbour two-sample test (the forgiving C2ST).

    Reports ``c2st/nn`` (mean per-niche real-fraction; ~0.5 indistinguishable, ->0 separable, ->1
    memorisation), ``c2st/nn_std`` (its spread, to expose a bimodal split-slide), and
    ``c2st/nn_real_ref`` (the same statistic on real centroids, a self-calibration that should also
    be ~0.5 for a balanced pool).

    """
    p = f"{prefix}/" if prefix else ""
    p_gen, p_real = c2st_nn(
        real_x,
        real_pos,
        gen_x,
        gen_pos,
        spatial_k=spatial_k,
        max_n=max_n,
        seed=seed,
    )
    return {
        f"{p}c2st/nn": float(p_gen.mean()),
        f"{p}c2st/nn_std": float(p_gen.std()),
        f"{p}c2st/nn_real_ref": float(p_real.mean()),
    }
