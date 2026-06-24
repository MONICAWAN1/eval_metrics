"""One-shot generate + evaluate CLI: ``python -m paired_slides_eval.pipeline``.

Model-agnostic: pick a generator by registry name (e.g. ``nicheflow``) or a ``module.path:callable``
spec pointing at your own model, and forward any model-specific options with ``--gen-kwarg``. For
finer control, call :func:`paired_slides_eval.pipeline.run.run_pipeline` from Python.
"""

from __future__ import annotations

import argparse
import csv
import os

from paired_slides_eval.evaluate import ALL_GROUPS
from paired_slides_eval.pipeline.run import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate cells with a model (a generator) from a checkpoint and evaluate them."
    )
    ap.add_argument(
        "--generator",
        default="nicheflow",
        help="generator: a registry name (default 'nicheflow') or a 'module.path:callable' spec",
    )
    ap.add_argument("--source", required=True, help="source slide .h5ad (raw genes + coords)")
    ap.add_argument("--target", required=True, help="target slide .h5ad (to generate)")
    ap.add_argument("--checkpoint", required=True, help="trained model checkpoint (-> generator)")
    ap.add_argument(
        "--classifier", default=None, help="optional classifier .ckpt (enables the ct/* groups)"
    )
    ap.add_argument("--out", default=None, help="write a metric,value CSV here")
    ap.add_argument(
        "--generated_out", default=None, help="also write the generated cells here (.h5ad / .npz)"
    )
    ap.add_argument(
        "--gen-kwarg",
        action="append",
        default=[],
        dest="gen_kwargs",
        metavar="KEY=VALUE",
        help="extra option forwarded to the generator, repeatable (e.g. for nicheflow: "
        "--gen-kwarg n_pcs=50 --gen-kwarg cell_type_column=class --gen-kwarg radius=0.15)",
    )
    ap.add_argument("--groups", nargs="+", default=None, choices=ALL_GROUPS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from paired_slides_eval.generate import _coerce, write_generated

    kwargs = {}
    for item in args.gen_kwargs:
        if "=" not in item:
            ap.error(f"--gen-kwarg expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        kwargs[key] = _coerce(value)

    groups = tuple(args.groups) if args.groups else ALL_GROUPS
    res = run_pipeline(
        args.source,
        args.target,
        args.checkpoint,
        generator=args.generator,     # resolved by name / dotted path inside run_pipeline
        classifier=args.classifier,
        groups=groups,
        seed=args.seed,
        **kwargs,
    )

    if args.generated_out:
        write_generated(res.generated, args.generated_out)

    metrics = dict(res.metrics)
    skipped = metrics.pop("_skipped", [])
    notes = metrics.pop("_notes", [])
    rows = sorted(metrics.items())
    print(f"target: {res.target.x.shape[0]} cells, {res.target.x.shape[1]} features")
    if res.generated.x.ndim == 3:
        print(f"generated: {res.generated.x.shape[0]} niches x {res.generated.x.shape[1]} points")
    else:
        g = res.generated
        print(f"generated: {g.x.shape[0]} cells, {g.x.shape[1]} feats (flat slide)")
    for k, v in rows:
        print(f"{k:24s} {v:.4f}")
    if notes:
        print("notes:", "; ".join(notes))
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
