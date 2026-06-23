"""CLI for the full pipeline: ``python -m nicheflow_eval.pipeline --source ... --target ...``."""

from __future__ import annotations

import argparse
import csv
import os

from nicheflow_eval.evaluate import ALL_GROUPS
from nicheflow_eval.pipeline.run import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate cells from a NicheFlow checkpoint and evaluate them, from raw AnnData."
    )
    ap.add_argument("--source", required=True, help="source slide .h5ad (raw genes+coords)")
    ap.add_argument("--target", required=True, help="target slide .h5ad (to generate)")
    ap.add_argument("--checkpoint", required=True, help="trained flow checkpoint")
    ap.add_argument("--classifier", default=None, help="held-out slide .h5ad for the classifier")
    ap.add_argument("--classifier_ckpt", default=None, help="load a trained classifier instead")
    ap.add_argument("--out", default=None, help="write a metric,value CSV here")
    ap.add_argument("--generated_out", default=None, help="write the generated cells .h5ad here")
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--cell_type_column", default="class")
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--dx", type=float, default=0.15)
    ap.add_argument("--dy", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--solver", default="euler")
    ap.add_argument("--variant", default="cfm", choices=["cfm", "vfm"])
    ap.add_argument("--n_slices", type=int, default=None, help="backbone ohe_dim (default #slides)")
    ap.add_argument("--groups", nargs="+", default=None, choices=ALL_GROUPS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    groups = tuple(args.groups) if args.groups else ALL_GROUPS
    res = run_pipeline(
        args.source,
        args.target,
        args.checkpoint,
        classifier_h5ad=args.classifier,
        classifier_ckpt=args.classifier_ckpt,
        n_pcs=args.n_pcs,
        cell_type_column=args.cell_type_column,
        radius=args.radius,
        dx=args.dx,
        dy=args.dy,
        device=args.device,
        num_steps=args.num_steps,
        solver=args.solver,
        variant=args.variant,
        n_slices=args.n_slices,
        groups=groups,
        seed=args.seed,
        generated_out=args.generated_out,
    )

    metrics = dict(res.metrics)
    skipped = metrics.pop("_skipped", [])
    rows = sorted(metrics.items())
    print(f"target: {res.target.x.shape[0]} cells, {res.target.x.shape[1]} features")
    print(f"generated: {res.generated.x.shape[0]} niches x {res.generated.x.shape[1]} points")
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


if __name__ == "__main__":
    main()
