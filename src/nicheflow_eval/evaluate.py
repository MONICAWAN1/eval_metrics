"""Top-level entry point: run the whole evaluation suite on one (target, generated) pair.

``evaluate`` returns a flat ``{prefix/group/metric: value}`` dict matching the columns the
NicheFlow result CSVs use, so existing reporting code keeps working. Each metric group is
optional and is skipped (with a note in the returned ``_skipped`` list) when its inputs are
absent — e.g. the classifier groups need a classifier + paired niches, regression needs matched
ground truth.

This is the **standalone** path: it takes a real ``TargetSlide`` and the model's
``GeneratedNiches`` (both built from AnnData) and never touches the flow model. To go all the way
from a checkpoint + raw slides, see :func:`nicheflow_eval.pipeline.run.run_pipeline`.
"""

from __future__ import annotations

from nicheflow_eval.contract import GeneratedNiches, GeneratedSlide, TargetSlide
from nicheflow_eval.data.anndata import read_anndata
from nicheflow_eval.metrics.c2st import c2st_metrics
from nicheflow_eval.metrics.classifier_gap import classifier_accuracy_gap
from nicheflow_eval.metrics.concordance import cell_type_concordance
from nicheflow_eval.metrics.distances import point_to_shape, regression_metrics, shape_to_point
from nicheflow_eval.metrics.distribution import distribution_distance
from nicheflow_eval.metrics.morans import morans_compare

ALL_GROUPS = (
    "regression",
    "psd",
    "spd",
    "distribution",
    "c2st",
    "moran",
    "concordance",
    "ct_gap",
)


def evaluate(
    target: TargetSlide,
    generated: GeneratedNiches | GeneratedSlide,
    *,
    classifier=None,
    classifier_spatial: bool = True,
    classifier_n_neighbors: int | None = None,
    groups: tuple[str, ...] = ALL_GROUPS,
    prefix: str = "test",
    seed: int = 0,
    c2st_max_n: int = 2000,
    c2st_n_folds: int = 5,
    c2st_n_perm: int = 0,
    mmd_max_n: int = 2000,
    ot_max_n: int = 4000,
    moran_n_neighs: int = 6,
) -> dict:
    """Compute every applicable metric for ``generated`` vs. ``target`` and return a flat dict.

    Groups: ``regression`` (needs ``generated.gt_*``), ``psd``/``spd``, ``distribution`` (MMD/EMD),
    ``c2st`` (per-cell joint + pos-only), ``moran`` (Moran's I over **all** generated cells),
    ``concordance`` (classifier agreement on generated vs paired-real niches) and ``ct_gap``
    (classifier accuracy gap real-vs-generated). The two classifier groups need a ``classifier``
    and the paired real niches ``generated.gt_*`` (``ct_gap`` also needs ``generated.gt_ct``).
    Pass a subset via ``groups`` to run only some.

    ``generated`` may be a :class:`~nicheflow_eval.contract.GeneratedNiches` (niche-shaped) or a
    flat :class:`~nicheflow_eval.contract.GeneratedSlide`. The label-free groups run on either; the
    niche groups (``regression``, ``concordance``, ``ct_gap``) need ``GeneratedNiches`` and are
    skipped for a flat slide.
    """
    out: dict[str, float] = {}
    skipped: list[str] = []

    if "regression" in groups:
        if (
            getattr(generated, "gt_x", None) is not None
            and getattr(generated, "gt_pos", None) is not None
        ):
            out.update(
                regression_metrics(
                    generated.x, generated.gt_x, generated.pos, generated.gt_pos, prefix=prefix
                )
            )
        else:
            skipped.append(
                "regression (needs niche-shaped `GeneratedNiches` with matched `gt_x`/`gt_pos`)"
            )

    if "psd" in groups:
        out.update(point_to_shape(generated.flat_pos, target.pos, prefix=prefix))

    if "spd" in groups:
        out.update(shape_to_point(generated.flat_pos, target.pos, prefix=prefix))

    if "distribution" in groups:
        out.update(
            distribution_distance(
                target.x,
                target.pos,
                generated.flat_x,
                generated.flat_pos,
                prefix=prefix,
                mmd_max_n=mmd_max_n,
                ot_max_n=ot_max_n,
                seed=seed,
            )
        )

    if "c2st" in groups:
        out.update(
            c2st_metrics(
                target.x,
                target.pos,
                generated.flat_x,
                generated.flat_pos,
                prefix=prefix,
                max_n=c2st_max_n,
                n_folds=c2st_n_folds,
                n_perm=c2st_n_perm,
                seed=seed,
            )
        )

    if "moran" in groups:
        # Moran's I over ALL generated cells (we generate them all) vs the full real slide.
        real_x, real_pos = target.moran_grid
        out.update(
            morans_compare(
                generated.flat_x,
                generated.flat_pos,
                real_x,
                real_pos,
                prefix=prefix,
                n_neighs=moran_n_neighs,
                seed=seed,
            )
        )

    if "concordance" in groups:
        if (
            classifier is not None
            and getattr(generated, "gt_x", None) is not None
            and getattr(generated, "gt_pos", None) is not None
        ):
            out.update(
                cell_type_concordance(
                    generated.x,
                    generated.pos,
                    generated.gt_x,
                    generated.gt_pos,
                    classifier,
                    prefix=prefix,
                    spatial=classifier_spatial,
                    n_neighbors=classifier_n_neighbors,
                    n_classes=target.n_classes,
                )
            )
        else:
            skipped.append(
                "concordance (needs `classifier` and paired real niches `generated.gt_x/gt_pos`)"
            )

    if "ct_gap" in groups:
        if (
            classifier is not None
            and getattr(generated, "gt_x", None) is not None
            and getattr(generated, "gt_pos", None) is not None
            and getattr(generated, "gt_ct", None) is not None
        ):
            out.update(
                classifier_accuracy_gap(
                    generated.x,
                    generated.pos,
                    generated.gt_x,
                    generated.gt_pos,
                    generated.gt_ct,
                    classifier,
                    prefix=prefix,
                    spatial=classifier_spatial,
                    n_neighbors=classifier_n_neighbors,
                )
            )
        else:
            skipped.append(
                "ct_gap (needs `classifier`, paired niches `generated.gt_x/gt_pos` and true "
                "centroid labels `generated.gt_ct`)"
            )

    out["_skipped"] = skipped
    return out


def build_spatial_classifier(
    ckpt_path: str,
    input_dim: int,
    output_dim: int,
    *,
    hidden_dim: int = 64,
    num_heads: int = 4,
    coord_dim: int = 2,
    mask_centroid: bool = True,
):
    """Reconstruct the spatial SetTransformer classifier and load a checkpoint into it.

    Net hyperparameters must match what was trained; ``load_spatial_classifier`` attaches the
    training ``n_neighbors`` so the classifier metrics build identically sized niches.
    """
    import torch

    from nicheflow_eval.classifier.nets import SpatialCTClassifierNet
    from nicheflow_eval.metrics._common import load_spatial_classifier

    clf = SpatialCTClassifierNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        coord_dim=coord_dim,
        num_heads=num_heads,
        mask_centroid=mask_centroid,
    )
    load_spatial_classifier(clf, torch.load(ckpt_path, map_location="cpu"))
    return clf


def _load_generated(path: str, *, niche_key: str = "niche_id") -> GeneratedNiches | GeneratedSlide:
    """Load generated cells, auto-detecting niche-shaped vs flat.

    ``.h5ad``: niche-shaped if ``obs[niche_key]`` is present, else a flat ``GeneratedSlide``.
    ``.npz``: niche-shaped if ``x`` is 3-D ``(B, N, D)`` (optionally with ``gt_x``/``gt_pos``/
    ``gt_ct``), else a flat ``GeneratedSlide`` from 2-D ``x``/``pos``.
    """
    if str(path).endswith(".h5ad"):
        adata = read_anndata(path)
        if niche_key in adata.obs:
            return GeneratedNiches.from_anndata(adata, niche_key=niche_key)
        return GeneratedSlide.from_anndata(adata)
    import numpy as np

    npz = np.load(path)
    if npz["x"].ndim == 3:
        extra = {k: npz[k] for k in ("gt_x", "gt_pos", "gt_ct") if k in npz}
        return GeneratedNiches(x=npz["x"], pos=npz["pos"], **extra)
    return GeneratedSlide(x=npz["x"], pos=npz["pos"])


def _main() -> None:
    import argparse
    import csv
    import os

    ap = argparse.ArgumentParser(
        description="Run the nicheflow-eval metric suite on a (target slide, generated cells) pair."
    )
    ap.add_argument("--target", required=True, help="the target slide .h5ad (raw genes + coords)")
    ap.add_argument(
        "--generated",
        required=True,
        help="generated cells. Niche-shaped: .npz with x (B,N,P), pos (B,N,P) "
        "[+ gt_x/gt_pos/gt_ct] or an .h5ad with obs['niche_id']. Flat (whole-slide): .npz with "
        "2-D x/pos, or an .h5ad with X + obsm['spatial'] (niche metrics are then skipped).",
    )
    ap.add_argument(
        "--classifier", default=None, help="optional classifier .ckpt (enables the ct/* groups)"
    )
    ap.add_argument("--out", default=None, help="optional path to write a metric,value CSV")
    ap.add_argument(
        "--expr_key", default=None, help="obsm/layers key for expression (default: adata.X)"
    )
    ap.add_argument("--spatial_key", default="spatial", help="obsm key for coordinates")
    ap.add_argument("--ct_key", default=None, help="obs column with cell types")
    ap.add_argument("--timepoint_key", default=None, help="obs column identifying the slide")
    ap.add_argument("--timepoint", default=None, help="slide id to keep (with --timepoint_key)")
    ap.add_argument(
        "--groups",
        nargs="+",
        default=None,
        choices=ALL_GROUPS,
        help=f"subset of metric groups to run (default: all -> {', '.join(ALL_GROUPS)})",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n_pcs", type=int, default=None, help="fit a PCA on the target to n PCs and project"
    )
    # Spatial classifier net hyperparameters (must match training; only used with --classifier).
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--coord_dim", type=int, default=2)
    ap.add_argument("--no_mask_centroid", action="store_true", help="ablation: keep the centroid")
    args = ap.parse_args()

    target = TargetSlide.from_anndata(
        args.target,
        timepoint=args.timepoint,
        expr_key=args.expr_key,
        spatial_key=args.spatial_key,
        ct_key=args.ct_key,
        timepoint_key=args.timepoint_key,
        n_pcs=args.n_pcs,
    )
    generated = _load_generated(args.generated).project(target.pca)

    classifier = None
    if args.classifier is not None:
        classifier = build_spatial_classifier(
            args.classifier,
            target.x.shape[1],
            target.n_classes,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            coord_dim=args.coord_dim,
            mask_centroid=not args.no_mask_centroid,
        )

    groups = tuple(args.groups) if args.groups else ALL_GROUPS
    res = evaluate(target, generated, classifier=classifier, groups=groups, seed=args.seed)

    skipped = res.pop("_skipped")
    rows = sorted(res.items())
    print(f"target: {target.x.shape[0]} cells, {target.x.shape[1]} features")
    if generated.x.ndim == 3:
        print(f"generated: {generated.x.shape[0]} niches x {generated.x.shape[1]} points")
    else:
        print(f"generated: {generated.x.shape[0]} cells, {generated.x.shape[1]} feats (flat slide)")
    for k, v in rows:
        print(f"{k:24s} {v:.4f}")
    if skipped:
        print("skipped:", "; ".join(skipped))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            wcsv = csv.writer(fh)
            wcsv.writerow(["metric", "value"])
            wcsv.writerows(rows)
        print(f"\nsaved {args.out}")


# Usage: python -m nicheflow_eval.evaluate --target target.h5ad --generated generated.npz
if __name__ == "__main__":
    _main()
