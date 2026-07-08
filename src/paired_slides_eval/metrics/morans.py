"""Per-PC Moran's I (spatial autocorrelation) via squidpy.

Moran's I measures whether a feature is spatially *clustered* (nearby cells have similar
values): ~+1 clustered, ~0 random, <0 dispersed. ``morans_i`` is the framework-free kernel (one
I per feature column); ``morans_compare`` scores a generated slide against the real slide by
comparing the two per-PC I vectors, variance-weighted by the real per-PC variance.

Because the model generates *every* cell of each niche (not just centroids), Moran's I is
computed over **all** generated cells pooled into one cloud, compared against the full real
target slide — no density-matched grid subsample is involved.

"""

from __future__ import annotations

import logging

import numpy as np

from paired_slides_eval.metrics._common import weighted_pearson


def morans_i(
    features: np.ndarray,
    positions: np.ndarray,
    n_neighs: int = 6,
    seed: int = 0,
) -> np.ndarray:
    """Moran's I per feature (column of ``features``) over a kNN spatial graph.

    Args:
        features: (N, F) feature matrix (e.g. PCA expression); one Moran's I per column.
        positions: (N, 2) spatial coordinates.
        n_neighs: neighbours for squidpy's generic kNN spatial graph (row-normalised weights).
    Returns:
        (F,) array of Moran's I, one per feature, in column order.

    """
    import anndata as ad
    import squidpy as sq

    logging.getLogger("squidpy").setLevel(logging.WARNING)

    n_features = features.shape[1]
    names = [f"PC{i}" for i in range(n_features)]
    adata = ad.AnnData(X=np.asarray(features, dtype=np.float32))
    adata.var_names = names
    adata.obsm["spatial"] = np.asarray(positions, dtype=np.float32)

    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n_neighs)
    sq.gr.spatial_autocorr(adata, mode="moran", genes=names, seed=seed, show_progress_bar=False)
    return adata.uns["moranI"].loc[names, "I"].to_numpy()


def _dedupe(pos: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Upsampling can repeat grid centroids -> exact duplicate generations; drop them so the kNN
    # graph has no zero-distance self-duplicates.
    _, keep = np.unique(pos, axis=0, return_index=True)
    return pos[keep], x[keep]


def morans_compare(
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    real_x: np.ndarray,
    real_pos: np.ndarray,
    *,
    prefix: str = "",
    n_neighs: int = 6,
    weight: str = "variance",
    seed: int = 0,
) -> dict[str, float]:
    """Compare per-PC Moran's I of all generated cells vs the full real target
    slide.

    ``gen_*`` are all generated cells pooled into one cloud (every cell of every niche, not just
    centroids). ``real_*`` are the real target slide's cells. PCs are weighted by the real per-PC
    variance (``"variance"``; also ``"uniform"`` / ``"sqrt"``).

    """
    p = f"{prefix}/" if prefix else ""
    gen_pos, gen_x = _dedupe(np.asarray(gen_pos), np.asarray(gen_x))
    real_x = np.asarray(real_x)
    real_pos = np.asarray(real_pos)

    i_gen = morans_i(gen_x, gen_pos, n_neighs=n_neighs, seed=seed)
    i_real = morans_i(real_x, real_pos, n_neighs=n_neighs, seed=seed)

    if weight == "uniform":
        w = np.ones(real_x.shape[1])
    else:
        var = real_x.var(axis=0)
        w = np.sqrt(var) if weight == "sqrt" else var
    wn = w / w.sum()

    return {
        f"{p}moran/mae": float((wn * np.abs(i_real - i_gen)).sum()),
        f"{p}moran/corr": weighted_pearson(i_real, i_gen, w),
        f"{p}moran/real_mean": float((wn * i_real).sum()),
        f"{p}moran/gen_mean": float((wn * i_gen).sum()),
    }
