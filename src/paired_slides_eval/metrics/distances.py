"""Pointwise regression metrics for matched generated-vs-ground-truth cells.

* regression (``x/*``, ``pos/*``): per matched cell, generated vs. ground-truth target — needs
  the matched target microenvironment, so it is only computed when ground truth is supplied.

"""

from __future__ import annotations

import numpy as np


def regression_metrics(
    gen_x: np.ndarray,
    gt_x: np.ndarray,
    gen_pos: np.ndarray,
    gt_pos: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Pointwise MSE/MAE of generated vs. matched ground-truth, for expression
    and coordinates.

    All four arrays are matched cell-for-cell (e.g. flattened ``(B, N, D)`` niches where the
    generated niche ``b`` was transported from the same source as the GT niche ``b``).

    """
    p = f"{prefix}/" if prefix else ""
    gx, gtx = np.asarray(gen_x), np.asarray(gt_x)
    gp, gtp = np.asarray(gen_pos), np.asarray(gt_pos)
    return {
        f"{p}x/mse": float(np.mean((gx - gtx) ** 2)),
        f"{p}x/mae": float(np.mean(np.abs(gx - gtx))),
        f"{p}pos/mse": float(np.mean((gp - gtp) ** 2)),
        f"{p}pos/mae": float(np.mean(np.abs(gp - gtp))),
    }
