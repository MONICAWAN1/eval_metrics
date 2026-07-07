"""Shared helpers for the metric modules.

These were duplicated three ways in the source repo (in ``flow_matching.py``,
``metric_sensitivity_test.py``, and ``diagnose_gcn_collapse.py``); here they live once. Every
function operates on plain NumPy arrays so the metrics stay framework-light.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
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


def knn_indices(coords: np.ndarray, k: int, *, centroids: np.ndarray | None = None) -> np.ndarray:
    """``(B, k+1)`` KNN index array: each centroid's self (distance 0) at column 0, then its ``k``
    nearest neighbours by Euclidean distance on ``coords``.

    ``k`` is the number of **neighbours** (so the niche has ``k + 1`` points); it is clamped to the
    available neighbours (``len(coords) - 1``). ``centroids`` selects the query cells (default: all).
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    k_eff = int(min(k, n - 1))
    tree = cKDTree(coords)
    query = coords if centroids is None else coords[np.asarray(centroids)]
    _, idx = tree.query(query, k=k_eff + 1)
    return np.asarray(idx).reshape(len(query), k_eff + 1)


def build_knn_point_set(
    coords: np.ndarray, expr: np.ndarray, k: int, *, centroids: np.ndarray | None = None
) -> np.ndarray:
    """Expression-only KNN microenvironments ``(B, k+1, n_feat)``, the centroid at index 0.

    Each centroid + its ``k`` nearest neighbours (by ``coords``), gathering only the **expression**
    rows from ``expr`` — coordinates pick membership, then are dropped (the classifier is
    coordinate-blind). The single KNN-niche builder shared by the training dataset
    (:class:`~paired_slides_eval.classifier.dataset.SpatialH5ADCTDataset`) and the eval-time
    reconstruction (:func:`build_paired_niches_from_flat`).
    """
    idx = knn_indices(coords, k, centroids=centroids)
    return np.asarray(expr)[idx]


def build_microenv_points(x: np.ndarray, pos: np.ndarray, k: int | None) -> np.ndarray:
    """Expression-only point set from a **pre-assembled** microenvironment.

    ``x`` / ``pos`` are ``(n_microenvs, n_points, *)`` with the centroid at point 0. Keeps the
    centroid + its ``k`` nearest *other* points (by distance to the centroid) and returns the
    **expression** only ``(n_microenvs, k+1, n_feat)``; coordinates pick the k nearest, then are
    dropped. ``k`` is the number of neighbours (niche size ``k+1``); ``None`` keeps all points.

    Unlike :func:`build_knn_point_set` (which builds niches from a flat cloud), this sub-selects from
    an already-grouped microenvironment — e.g. NicheFlow's generation niches, which carry more points
    than the classifier's ``k``.
    """
    x = np.asarray(x)
    if k is not None and pos.shape[1] > k + 1:
        rel = np.linalg.norm(pos - pos[:, :1, :], axis=-1)  # (n_microenvs, n_points)
        idx = np.argsort(rel, axis=-1)[:, : k + 1]  # centroid (dist 0) first, then k nearest
        x = np.take_along_axis(x, idx[..., None], axis=1)
    return x


def build_paired_niches_from_flat(
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    real_x: np.ndarray,
    real_pos: np.ndarray,
    k: int,
    *,
    real_ct: np.ndarray | None = None,
    centroid_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Assemble paired microenvironments from a **flat** generated slide via geometry.

    A whole-slide model emits cells with no niche structure, so the niche/classifier metrics have
    no ``(B, N, D)`` microenvironments to compare. This rebuilds them the same way the classifier
    was trained (:class:`~paired_slides_eval.classifier.dataset.SpatialH5ADCTDataset`): a niche is a
    centroid cell plus its ``k`` nearest spatial neighbours (KNN), centroid first.

    For each chosen generated centroid we build:

    * the **generated** niche — the centroid + its ``k`` nearest *generated* neighbours;
    * the **paired real** niche — the real cell nearest to that centroid (by coordinates) + that
      real cell's ``k`` nearest *real* neighbours. This is the geometric stand-in for the
      transport pairing a niche-aware model supplies via ``gt_x``/``gt_pos``.

    The generated and real slides must live in the **same coordinate frame** (as they already must
    for ``psd``/``spd``/``moran``) so the nearest-real match is meaningful.

    Args:
        gen_x / gen_pos: flat generated cells ``(N_gen, n_feat)`` / ``(N_gen, coord)``.
        real_x / real_pos: the real target cells ``(N_real, n_feat)`` / ``(N_real, coord)``.
        k: number of **neighbours** per niche (niche size ``k + 1``) — the classifier's KNN ``k``.
            Clamped to the available cells on each side.
        real_ct: ``(N_real,)`` integer cell-type labels of the real cells; if given, the matched
            real centroid's label is returned as ``gt_ct`` (enables the accuracy-gap metric).
        centroid_indices: which generated cells to use as centroids; default all of them.

    Returns:
        ``(gen_niche_x, gen_niche_pos, gt_x, gt_pos, gt_ct)`` — the first four ``(B, k+1, D)`` with
        the centroid at point 0; ``gt_ct`` is ``(B,)`` or ``None`` when ``real_ct`` is absent. The
        ``*_pos`` are kept only for a downstream sub-KNN; the classifier reads expression only.
    """
    gen_x = np.asarray(gen_x)
    gen_pos = np.asarray(gen_pos)
    real_x = np.asarray(real_x)
    real_pos = np.asarray(real_pos)

    if len(gen_pos) < 1 or len(real_pos) < 1:
        raise ValueError("Need at least one generated and one real cell to build niches.")

    centroids = (
        np.arange(len(gen_pos)) if centroid_indices is None else np.asarray(centroid_indices)
    )

    # Generated niche: each centroid + its k nearest generated cells (centroid at column 0).
    gen_nbr = knn_indices(gen_pos, k, centroids=centroids)  # (B, k+1)

    # Paired real niche: the real cell nearest each centroid, then that real cell's k nearest cells.
    _, r0 = cKDTree(real_pos.astype(np.float64)).query(gen_pos[centroids], k=1)
    real_nbr = knn_indices(real_pos, k, centroids=r0)  # (B, k+1)

    gen_niche_x = gen_x[gen_nbr]
    gen_niche_pos = gen_pos[gen_nbr]
    gt_x = real_x[real_nbr]
    gt_pos = real_pos[real_nbr]
    gt_ct = None if real_ct is None else np.asarray(real_ct)[r0].astype(np.int64)
    return gen_niche_x, gen_niche_pos, gt_x, gt_pos, gt_ct


def build_paired_niches_from_flat_fixed_centroids(
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    real_x: np.ndarray,
    real_pos: np.ndarray,
    k: int,
    *,
    target_centroid_indices: np.ndarray,
    real_ct: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Assemble paired niches for a flat slide using fixed target centroids plus OT assignment.

    This is the cross-model-comparable flat-slide path. NicheFlow already generates at the
    preprocessed target grid centroids (``subsampled_timepoint_idx``), so its ``gt_*`` pairing is
    anchored to those target cells. A flat whole-slide model such as OT-CFM has no supplied
    one-to-one target pairing, so we hold the **same target centroids** fixed and solve a balanced
    linear assignment from those centroids to generated cells in the shared coordinate frame. Each
    target centroid gets one unique generated centroid, then both sides build their own spatial KNN
    neighbourhoods around the matched centroid.

    Args:
        gen_x / gen_pos: flat generated cells ``(N_gen, n_feat)`` / ``(N_gen, coord)``.
        real_x / real_pos: the real target cells ``(N_real, n_feat)`` / ``(N_real, coord)``.
        k: number of **neighbours** per niche (niche size ``k + 1``), clamped per side.
        target_centroid_indices: local indices of fixed real target centroids to evaluate.
        real_ct: optional ``(N_real,)`` true labels; labels at the fixed target centroids become
            ``gt_ct``.

    Returns:
        ``(gen_niche_x, gen_niche_pos, gt_x, gt_pos, gt_ct)`` in target-centroid order.
    """
    gen_x = np.asarray(gen_x)
    gen_pos = np.asarray(gen_pos)
    real_x = np.asarray(real_x)
    real_pos = np.asarray(real_pos)
    target_centroids = np.asarray(target_centroid_indices, dtype=np.int64)

    if len(gen_pos) < 1 or len(real_pos) < 1:
        raise ValueError("Need at least one generated and one real cell to build niches.")
    if len(target_centroids) < 1:
        raise ValueError("Need at least one fixed target centroid to build niches.")
    if np.any(target_centroids < 0) or np.any(target_centroids >= len(real_pos)):
        raise ValueError("target_centroid_indices contains indices outside the real target slide.")
    if len(gen_pos) < len(target_centroids):
        raise ValueError(
            "OT pairing needs at least as many generated cells as fixed target centroids "
            f"({len(gen_pos)} generated < {len(target_centroids)} target centroids)."
        )

    target_pos = real_pos[target_centroids].astype(np.float64)
    gen_pos64 = gen_pos.astype(np.float64)
    diff = target_pos[:, None, :] - gen_pos64[None, :, :]
    cost = np.einsum("...d,...d->...", diff, diff)
    row_ind, gen_centroids = linear_sum_assignment(cost)
    target_centroids = target_centroids[row_ind]

    gen_nbr = knn_indices(gen_pos, k, centroids=gen_centroids)
    real_nbr = knn_indices(real_pos, k, centroids=target_centroids)

    gen_niche_x = gen_x[gen_nbr]
    gen_niche_pos = gen_pos[gen_nbr]
    gt_x = real_x[real_nbr]
    gt_pos = real_pos[real_nbr]
    gt_ct = None if real_ct is None else np.asarray(real_ct)[target_centroids].astype(np.int64)
    return gen_niche_x, gen_niche_pos, gt_x, gt_pos, gt_ct


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

    This package's spatial classifier wraps the net as ``self.net`` inside the LightningModule, so
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
    (:class:`~paired_slides_eval.classifier.task.CellTypeClassification`) records in the checkpoint.
    :func:`~paired_slides_eval.metrics.concordance.cell_type_concordance` reads this so eval builds
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
