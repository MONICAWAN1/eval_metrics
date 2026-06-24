"""Distribution-distance metrics: MMD and Wasserstein/EMD (W1/W2).

``mmd2_rbf`` and ``ot_distance`` are the framework-free kernels (mixture-of-RBF squared MMD and
exact-EMD Wasserstein-`power`); ``distribution_distance`` is the convenience wrapper that scores
the generated cloud against the real target cloud separately for expression (``x``) and
coordinates (``pos``) — the population-level "is the generated distribution right?" test.
"""

from __future__ import annotations

import numpy as np


def mmd2_rbf(
    x: np.ndarray,
    y: np.ndarray,
    sigma_list: tuple[float, ...] | None = None,
    bandwidth_mults: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    biased: bool = False,
    max_n: int = 2000,
    seed: int = 0,
) -> float:
    """Mixture-of-RBF squared MMD between two sample sets.

    Pool X and Y, build the full Gram matrix, sum an RBF kernel over several bandwidths (median
    heuristic by default), and form MMD^2 from the XX / YY / XY blocks. Pairing-free. Unbiased
    U-statistic by default (the value can dip slightly negative when distributions match; that
    is expected, do not clip). Both sets are subsampled to ``max_n`` rows (seeded).
    """
    import torch

    rng = np.random.default_rng(seed)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]

    X = torch.as_tensor(x, dtype=torch.float32)
    Y = torch.as_tensor(y, dtype=torch.float32)
    n, m = X.shape[0], Y.shape[0]

    Z = torch.cat((X, Y), dim=0)
    ZZT = Z @ Z.t()
    diag = torch.diag(ZZT).unsqueeze(1)
    d2 = (diag - 2.0 * ZZT + diag.t()).clamp_min(0.0)  # ||z_i - z_j||^2

    if sigma_list is None:
        off = d2[~torch.eye(n + m, dtype=torch.bool)]
        med = float(off.median())
        med = med if med > 0 else 1.0
        sigma2_list = [med * mult for mult in bandwidth_mults]
    else:
        sigma2_list = [float(s) ** 2 for s in sigma_list]

    K = torch.zeros_like(d2)
    for s2 in sigma2_list:
        K = K + torch.exp(-d2 / (2.0 * s2))

    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    if biased:
        mmd2 = Kxx.sum() / (n * n) + Kyy.sum() / (m * m) - 2.0 * Kxy.sum() / (n * m)
    else:
        sum_xx = Kxx.sum() - torch.diagonal(Kxx).sum()
        sum_yy = Kyy.sum() - torch.diagonal(Kyy).sum()
        mmd2 = sum_xx / (n * (n - 1)) + sum_yy / (m * (m - 1)) - 2.0 * Kxy.sum() / (n * m)
    return float(mmd2)


def ot_distance(
    x: np.ndarray,
    y: np.ndarray,
    method: str | None = "exact",
    reg: float = 0.05,
    power: int = 2,
    max_n: int = 4000,
    seed: int = 0,
) -> float:
    """Wasserstein-`power` (EMD) distance between two sample sets (Euclidean cost).

    Uniform marginals, cost M = ||x_i - y_j|| (squared when power==2), solved with POT's exact
    EMD (``emd2``) or entropic Sinkhorn. For power==2 the sqrt is taken, so the return is the
    true W2 distance (not W2^2). Sets are subsampled to ``max_n`` rows (seeded).
    """
    import ot as pot
    import torch

    assert power == 1 or power == 2
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], max_n, replace=False)]

    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        from functools import partial

        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x.shape[0]), pot.unif(y.shape[0])
    M = torch.cdist(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = np.sqrt(ret)
    return float(ret)


def distribution_distance(
    real_x: np.ndarray,
    real_pos: np.ndarray,
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    *,
    prefix: str = "",
    mmd_max_n: int = 2000,
    ot_max_n: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """MMD^2 and Wasserstein-1/2 of generated vs. real, scored separately for ``x`` and ``pos``.

    Inputs are flat clouds: ``real_*`` are all real target cells, ``gen_*`` are all generated
    cells (flatten the ``(B, N, D)`` niches first). The real sample is passed first so it acts
    as the reference where the ground cost / kernel is symmetric anyway.
    """
    p = f"{prefix}/" if prefix else ""
    return {
        f"{p}mmd2/x": mmd2_rbf(real_x, gen_x, max_n=mmd_max_n, seed=seed),
        f"{p}mmd2/pos": mmd2_rbf(real_pos, gen_pos, max_n=mmd_max_n, seed=seed),
        f"{p}ot_w1/x": ot_distance(real_x, gen_x, power=1, max_n=ot_max_n, seed=seed),
        f"{p}ot_w1/pos": ot_distance(real_pos, gen_pos, power=1, max_n=ot_max_n, seed=seed),
        f"{p}ot_w2/x": ot_distance(real_x, gen_x, power=2, max_n=ot_max_n, seed=seed),
        f"{p}ot_w2/pos": ot_distance(real_pos, gen_pos, power=2, max_n=ot_max_n, seed=seed),
    }
