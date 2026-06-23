"""Top-level entry point: run the whole evaluation suite on one (target, generated) pair.

``evaluate`` returns a flat ``{prefix/group/metric: value}`` dict matching the columns the
NicheFlow result CSVs use, so existing reporting code keeps working. Each metric group is
optional and is skipped (with a note in the returned ``_skipped`` list) when its inputs are
absent — e.g. concordance needs a classifier + labels, regression needs matched ground truth.
"""

from __future__ import annotations

from nicheflow_eval.contract import GeneratedNiches, TargetSlide
from nicheflow_eval.metrics.c2st import c2st_metrics
from nicheflow_eval.metrics.concordance import cell_type_concordance
from nicheflow_eval.metrics.distances import point_to_shape, regression_metrics, shape_to_point
from nicheflow_eval.metrics.distribution import distribution_distance
from nicheflow_eval.metrics.morans import morans_compare

ALL_GROUPS = ("regression", "psd", "spd", "distribution", "c2st", "moran", "concordance")


def evaluate(
    target: TargetSlide,
    generated: GeneratedNiches,
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
    ``c2st`` (per-cell joint + pos-only), ``moran`` (Moran's I), ``concordance`` (needs a neutral
    ``classifier`` and the paired real niches ``generated.gt_*``). Pass a subset via ``groups`` to
    run only some.
    """
    out: dict[str, float] = {}
    skipped: list[str] = []

    if "regression" in groups:
        if generated.gt_x is not None and generated.gt_pos is not None:
            out.update(
                regression_metrics(
                    generated.x, generated.gt_x, generated.pos, generated.gt_pos, prefix=prefix
                )
            )
        else:
            skipped.append("regression (no matched ground-truth niches on `generated`)")

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
        grid_x, grid_pos = target.moran_grid
        out.update(
            morans_compare(
                generated.centroid_x,
                generated.centroid_pos,
                grid_x,
                grid_pos,
                prefix=prefix,
                n_neighs=moran_n_neighs,
                seed=seed,
            )
        )

    if "concordance" in groups:
        if classifier is not None and generated.gt_x is not None and generated.gt_pos is not None:
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

    out["_skipped"] = skipped
    return out


def _build_classifier(ckpt_path: str, input_dim: int, output_dim: int, args):
    """Reconstruct the spatial SetTransformer classifier and load a checkpoint into it.

    Net hyperparameters must match what was trained (mirrors the ``abca_spatial`` config defaults);
    ``load_spatial_classifier`` attaches the training ``n_neighbors`` so concordance matches it.
    """
    import torch

    from nicheflow_eval.classifier.nets import SpatialCTClassifierNet
    from nicheflow_eval.metrics._common import load_spatial_classifier

    clf = SpatialCTClassifierNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        coord_dim=args.coord_dim,
        num_heads=args.num_heads,
        mask_centroid=not args.no_mask_centroid,
    )
    load_spatial_classifier(clf, torch.load(ckpt_path, map_location="cpu"))
    return clf


def _main() -> None:
    import argparse
    import csv
    import os

    import numpy as np

    from nicheflow_eval.contract import GeneratedNiches, TargetSlide
    from nicheflow_eval.data import load_h5ad_dataset_dataclass

    ap = argparse.ArgumentParser(
        description="Run the nicheflow-eval metric suite on one (target slide, generated cells) pair."
    )
    ap.add_argument(
        "--target",
        required=True,
        help="the target slide's own preprocessing .pkl (e.g. target_abca.pkl) — NOT the "
        "concatenated source+target aligned pkl",
    )
    ap.add_argument(
        "--generated",
        required=True,
        help="generated .npz with arrays x (B,N,P), pos (B,N,P); optional gt_x, gt_pos",
    )
    ap.add_argument("--classifier", default=None, help="optional classifier .ckpt (enables the concordance group)")
    ap.add_argument("--out", default=None, help="optional path to write a metric,value CSV")
    ap.add_argument(
        "--timepoint",
        default=None,
        help="target slide id (default: the last in timepoints_ordered)",
    )
    ap.add_argument(
        "--groups",
        nargs="+",
        default=None,
        choices=ALL_GROUPS,
        help=f"subset of metric groups to run (default: all -> {', '.join(ALL_GROUPS)})",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_pcs", type=int, default=None, help="truncate expression to the first n PCs")
    # Spatial classifier net hyperparameters (must match training; only used with --classifier).
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--coord_dim", type=int, default=2)
    ap.add_argument("--no_mask_centroid", action="store_true", help="ablation: keep the centroid")
    args = ap.parse_args()

    ds = load_h5ad_dataset_dataclass(args.target)
    timepoint = args.timepoint or ds.timepoints_ordered[-1]
    target = TargetSlide.from_dataclass(ds, timepoint=timepoint, n_pcs=args.n_pcs)

    npz = np.load(args.generated)
    extra = {k: npz[k] for k in ("gt_x", "gt_pos") if k in npz}
    generated = GeneratedNiches(x=npz["x"], pos=npz["pos"], **extra)

    classifier = None
    if args.classifier is not None:
        classifier = _build_classifier(args.classifier, target.x.shape[1], target.n_classes, args)

    groups = tuple(args.groups) if args.groups else ALL_GROUPS
    res = evaluate(target, generated, classifier=classifier, groups=groups, seed=args.seed)

    skipped = res.pop("_skipped")
    rows = sorted(res.items())
    print(f"target '{timepoint}': {target.x.shape[0]} cells, {target.x.shape[1]} PCs")
    print(f"generated: {generated.x.shape[0]} niches x {generated.x.shape[1]} points")
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


# Usage: python -m nicheflow_eval.evaluate --target target.pkl --generated generated.npz
if __name__ == "__main__":
    _main()
