"""Geometric distance metrics: pointwise regression, point-to-shape, shape-to-point.

* regression (``x/*``, ``pos/*``): per matched cell, generated vs. ground-truth target — needs
  the matched target microenvironment, so it is only computed when ground truth is supplied.
* point-to-shape (``psd/*``): for each generated cell, distance to its nearest real cell
  (how close generations land to the real manifold).
* shape-to-point (``spd/*``): for each real cell, distance to its nearest generated cell
  (how well generations cover the real distribution).
"""

from __future__ import annotations

import numpy as np

from paired_slides_eval.metrics._common import nn_query


def regression_metrics(
    gen_x: np.ndarray,
    gt_x: np.ndarray,
    gen_pos: np.ndarray,
    gt_pos: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Pointwise MSE/MAE of generated vs. matched ground-truth, for expression and coordinates.

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


def point_to_shape(gen_pos: np.ndarray, real_pos: np.ndarray, *, prefix: str = "") -> dict[str, float]:
    """For each generated cell, distance to its nearest real cell (mean and max; lower better)."""
    p = f"{prefix}/" if prefix else ""
    _, dists = nn_query(gen_pos, real_pos)
    return {f"{p}psd/mean": float(dists.mean()), f"{p}psd/max": float(dists.max())}


def shape_to_point(gen_pos: np.ndarray, real_pos: np.ndarray, *, prefix: str = "") -> dict[str, float]:
    """For each real cell, distance to its nearest generated cell (mean and max; lower better)."""
    p = f"{prefix}/" if prefix else ""
    _, dists = nn_query(real_pos, gen_pos)
    return {f"{p}spd/mean": float(dists.mean()), f"{p}spd/max": float(dists.max())}
