"""UMAP overlay of generated vs target cells — a qualitative complement to the metric suite.

A single UMAP is fit **once on the target slide's shared whitened ``X_pca(50)``** and every model's
generated cells are projected through that frozen embedding, so all panels share one coordinate
frame and the target's structure is never distorted by a model. **Target** cells are coloured by
their true cell type (marker ``o``); **generated** cells are plotted in grey (marker ``^``), so each
panel shows whether a model's cells fall onto the target's populated cell-type regions.

Reads a preprocessed pair ``.pkl`` (the eval target, in ``X_pca(50)``) and each model's generated
``.h5ad`` — the same inputs ``evaluate`` consumes. Pure ``umap`` + ``matplotlib``.
"""

from __future__ import annotations

import numpy as np


def umap_compare(
    pair_pkl: str,
    models: list[tuple[str, str]],
    *,
    out_path: str | None = None,
    seed: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    point_size: float = 3.0,
    alpha: float = 0.5,
):
    """Fit UMAP on the target and overlay each model's generated cells; return the figure.

    Args:
        pair_pkl: preprocessed pair ``.pkl`` (the eval target, carrying ``X_pca`` + ``ct`` labels).
        models: ``[(name, generated.h5ad), ...]`` — one entry per model (a 2x2 grid holds up to 4).
        out_path: if given, save the figure there (e.g. ``reports/umap_compare.png``).
        seed / n_neighbors / min_dist: UMAP fit controls (``seed`` fixes the layout).
        point_size / alpha: scatter styling (lower both for dense "all cells" plots).
    """
    import matplotlib.pyplot as plt
    import umap
    from matplotlib.lines import Line2D

    from paired_slides_eval.contract import TargetSlide
    from paired_slides_eval.data.dataclass import load_h5ad_dataset_dataclass
    from paired_slides_eval.loaders import _load_generated

    ds = load_h5ad_dataset_dataclass(pair_pkl)
    ct_names = list(ds.ct_ordered)  # index -> cell-type name (the shared 20-class vocabulary)

    # Target: shared X_pca(50) + true integer cell-type labels (indices into ct_names).
    target = TargetSlide.from_dataclass(pair_pkl)
    target_x = np.asarray(target.x, dtype=np.float32)
    target_ct = np.asarray(target.ct, dtype=np.int64)

    # Fit UMAP ONCE on the target; every model is transformed through this frozen embedding.
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed, n_components=2
    ).fit(target_x)
    target_emb = reducer.embedding_

    # One colour per cell type (tab20 spans the 20-class vocabulary).
    cmap = plt.get_cmap("tab20")
    colors = {i: cmap(i % 20) for i in range(len(ct_names))}

    n = len(models)
    nrows, ncols = (2, 2) if n > 1 else (1, 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 7 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, (name, gen_path) in zip(axes, models, strict=False):
        try:
            gen = _load_generated(gen_path).project(target.pca)
            gen_x = np.asarray(gen.flat_x, dtype=np.float32)  # (N, 50) in the shared basis
        except (FileNotFoundError, ValueError) as exc:
            # e.g. an un-regenerated OT-CFM artifact still in its own low-dim PCA -> not comparable.
            reason = "not in shared X_pca(50)" if isinstance(exc, ValueError) else "file not found"
            msg = f"{name}\n\nskipped - {reason}\n(regenerate in the shared space, section 2b)"
            ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10, transform=ax.transAxes)
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        gen_emb = reducer.transform(gen_x)

        # Target = circles coloured by true cell type; generated = grey triangles on top.
        ax.scatter(
            target_emb[:, 0], target_emb[:, 1], s=point_size, alpha=alpha, linewidths=0,
            marker="o", c=[colors[c] for c in target_ct],
        )
        ax.scatter(
            gen_emb[:, 0], gen_emb[:, 1], s=point_size, alpha=alpha, linewidths=0,
            marker="^", color="0.55",
        )
        ax.set_title(f"{name}   (target n={len(target_x)}, gen n={len(gen_x)})")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:  # hide unused grid cells
        ax.axis("off")

    # Shared legend: the target/generated marker key + the target cell-type colours.
    present = sorted(set(target_ct.tolist()))
    ct_handles = [
        Line2D([], [], marker="s", linestyle="", markersize=7, color=colors[i], label=ct_names[i])
        for i in present
    ]
    shape_handles = [
        Line2D([], [], marker="o", linestyle="", markersize=7, color="0.3", label="target (by ct)"),
        Line2D([], [], marker="^", linestyle="", markersize=7, color="0.55", label="generated"),
    ]
    fig.legend(
        handles=shape_handles + ct_handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
        fontsize=8, frameon=False, title="markers / cell types",
    )
    fig.suptitle("UMAP: generated vs target (shared X_pca(50), fit on target)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.86, 0.97))

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved {out_path}")
    return fig
