"""Shared helpers for the metric modules.

These were duplicated three ways in the source repo (in ``flow_matching.py``,
``metric_sensitivity_test.py``, and ``diagnose_gcn_collapse.py``); here they live once. Every
function operates on plain NumPy arrays so the metrics stay framework-light.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def nn_query(query: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row of ``query`` find its nearest neighbour in ``ref`` (Euclidean).

    Returns ``(indices, distances)``, each shape ``(len(query),)``. Uses a KD-tree, so this is
    exact and scales to large clouds (the source used chunked ``torch.cdist``; the nearest
    neighbour is identical).
    """
    tree = cKDTree(np.asarray(ref, dtype=np.float64))
    dist, idx = tree.query(np.asarray(query, dtype=np.float64), k=1)
    return idx, dist


def build_microenv_points(
    x: np.ndarray, pos: np.ndarray, k: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Turn microenvironment point clouds into ``[expression | relative_position]`` point sets.

    ``x`` / ``pos`` are ``(n_microenvs, n_points, *)`` with the centroid at point 0. Positions
    are made relative to the centroid; if there are more than ``k`` points the ``k`` closest to
    the centroid are kept (the centroid, distance 0, is always included). Returns the point sets
    ``(n_microenvs, k, n_pcs + coord_dim)`` and the centroid positions ``(n_microenvs, coord)``.
    """
    centroid_pos = pos[:, :1, :]  # (n_microenvs, 1, coord)
    rel_pos = pos - centroid_pos  # (n_microenvs, n_points, coord)

    if k is not None and pos.shape[1] > k:
        dist = np.linalg.norm(rel_pos, axis=-1)  # (n_microenvs, n_points)
        idx = np.argsort(dist, axis=-1)[:, :k]  # (n_microenvs, k) closest first
        x = np.take_along_axis(x, idx[..., None], axis=1)
        rel_pos = np.take_along_axis(rel_pos, idx[..., None], axis=1)

    points = np.concatenate([x, rel_pos], axis=-1)
    return points, centroid_pos.squeeze(1)


def weighted_pearson(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation between vectors ``a`` and ``b`` with weights ``w``."""
    w = w / w.sum()
    ma, mb = float((w * a).sum()), float((w * b).sum())
    cov = float((w * (a - ma) * (b - mb)).sum())
    va, vb = float((w * (a - ma) ** 2).sum()), float((w * (b - mb) ** 2).sum())
    denom = (va * vb) ** 0.5
    return cov / denom if denom > 0 else float("nan")


def proportions(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Class-proportion histogram over ``n_classes`` integer labels (sums to 1)."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def kl_divergence(p_real: np.ndarray, p_gen: np.ndarray, eps: float = 1e-8) -> float:
    """KL(real || generated) between two proportion histograms."""
    return float((p_real * (np.log(p_real + eps) - np.log(p_gen + eps))).sum())


def total_variation(p_real: np.ndarray, p_gen: np.ndarray) -> float:
    """Total-variation distance between two proportion histograms (bounded [0, 1])."""
    return float(0.5 * np.abs(p_real - p_gen).sum())


def jensen_shannon(p_real: np.ndarray, p_gen: np.ndarray, eps: float = 1e-8) -> float:
    """Jensen-Shannon divergence between two proportion histograms (symmetric, bounded)."""
    m = 0.5 * (p_real + p_gen)
    return float(
        0.5 * (p_real * (np.log(p_real + eps) - np.log(m + eps))).sum()
        + 0.5 * (p_gen * (np.log(p_gen + eps) - np.log(m + eps))).sum()
    )


def subsample(arr: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly subsample rows of ``arr`` to at most ``max_n`` (seeded via ``rng``)."""
    if len(arr) > max_n:
        return arr[rng.choice(len(arr), max_n, replace=False)]
    return arr


def strip_module_prefix(state_dict: dict, prefix: str = "net.") -> dict:
    """Strip a leading ``prefix`` (e.g. ``net.``) from Lightning checkpoint state-dict keys.

    NicheFlow classifier checkpoints wrap the net as ``self.net`` inside the LightningModule, so
    the saved keys are ``net.<...>``. This returns only those keys, de-prefixed, ready for
    ``net.load_state_dict``.
    """
    out = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix):]] = value
    return out


def load_spatial_classifier(net, checkpoint: dict, prefix: str = "net."):
    """Load a spatial cell-type classifier from a Lightning ``checkpoint`` dict into ``net``.

    Loads the (de-prefixed) weights and attaches ``net.n_neighbors`` — the *effective*
    microenvironment size the classifier was trained on, which the training task
    (:class:`~nicheflow_eval.classifier.task.CellTypeClassification`) records in the checkpoint.
    :func:`~nicheflow_eval.metrics.concordance.cell_type_concordance` reads this so eval builds
    identically sized niches without the caller hardcoding ``n_neighbors``. Checkpoints predating
    this (no recorded value) load fine; ``n_neighbors`` is simply left unset and eval falls back
    to its default with a warning.

    ``checkpoint`` is an already-loaded checkpoint dict, e.g. ``torch.load(path, map_location=...)``.
    Returns ``net`` for convenience.
    """
    state_dict = checkpoint.get("state_dict", checkpoint)
    net.load_state_dict(strip_module_prefix(state_dict, prefix))
    n_neighbors = checkpoint.get("n_neighbors")
    if n_neighbors is not None:
        net.n_neighbors = int(n_neighbors)
    return net
