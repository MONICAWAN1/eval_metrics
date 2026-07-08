"""Command-line interface for the evaluation suite.

Run as ``python -m paired_slides_eval.cli`` (or, equivalently, ``python -m paired_slides_eval.evaluate``,
which forwards here). A thin argparse wrapper over
:func:`paired_slides_eval.evaluate.evaluate_files`.
"""

from __future__ import annotations

from paired_slides_eval.evaluate import ALL_GROUPS, evaluate_files


def main() -> None:
    import argparse
    import csv
    import os

    ap = argparse.ArgumentParser(
        description="Run the metric suite on a (target slide, generated cells) pair."
    )
    ap.add_argument(
        "--target",
        required=True,
        help="the target slide: a preprocessed-slide .pkl (scored in the shared whitened X_pca; "
        "--expr_key ignored)",
    )
    ap.add_argument(
        "--generated",
        required=True,
        help="generated cells. Niche-shaped: .npz with x (B,N,P), pos (B,N,P) "
        "[+ gt_x/gt_pos/gt_ct], an .h5ad with obs['niche_id'], or a .pkl (GenerationResult / dict "
        "of arrays). Flat (whole-slide): .npz with 2-D x/pos, or an .h5ad with X + "
        "obsm['spatial'] (niche metrics are then auto-built from geometry when a classifier or "
        "regressor is given).",
    )
    ap.add_argument(
        "--classifier", default=None, help="optional classifier .ckpt (enables the ct/* groups)"
    )
    ap.add_argument(
        "--regressor", default=None, help="optional expression-regressor .ckpt (enables recon)"
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
        "--coords",
        default="auto",
        choices=("auto", "standardize", "passthrough"),
        help="how generated coords are reconciled to a .pkl target's standardised frame: 'auto' "
        "(default — detect raw vs standardised and map accordingly), or force one. Replaces the old "
        "--standardize_coords flag.",
    )
    ap.add_argument(
        "--no_apply_lognorm",
        action="store_true",
        help="for a shared-PCA .pkl target: do NOT re-apply normalize_total+log1p to gene-space cells "
        "(set this if the generated expression is already log-normalised; see comparability_plan.md)",
    )
    ap.add_argument(
        "--ct_real_reference",
        default="paired",
        choices=("paired", "fixed"),
        help="how ct/acc_real is measured: 'paired' (default, model-dependent) or 'fixed' (a seeded "
        "model-independent sample of real target centroids — use for cross-model comparison tables)",
    )
    ap.add_argument(
        "--ct_real_n", type=int, default=2000, help="centroids sampled when --ct_real_reference fixed"
    )
    ap.add_argument(
        "--flat_pairing",
        default="fixed_target_ot",
        choices=("fixed_target_ot", "nearest_real"),
        help="for flat generated slides with classifier/recon metrics: 'fixed_target_ot' (default) "
        "uses the target's fixed evaluation centroids and OT-assigns them to generated cells; "
        "'nearest_real' uses the legacy generated-centroid nearest-real pairing",
    )
    ap.add_argument(
        "--recon_real_reference",
        default="fixed",
        choices=("paired", "fixed"),
        help="how recon/mse_real is measured: 'fixed' (default, model-independent) or 'paired'",
    )
    ap.add_argument(
        "--recon_real_n",
        type=int,
        default=2000,
        help="centroids sampled when --recon_real_reference fixed",
    )
    # Spatial classifier/regressor net hyperparameters (must match training; used with checkpoints).
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--no_mask_centroid", action="store_true", help="ablation: keep the centroid")
    args = ap.parse_args()

    res = evaluate_files(
        args.target,
        args.generated,
        ct_key=args.ct_key,
        classifier=args.classifier,
        regressor=args.regressor,
        classifier_kwargs=dict(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            mask_centroid=not args.no_mask_centroid,
        ),
        regressor_kwargs=dict(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            mask_centroid=not args.no_mask_centroid,
        ),
        groups=tuple(args.groups) if args.groups else ALL_GROUPS,
        expr_key=args.expr_key,
        spatial_key=args.spatial_key,
        timepoint=args.timepoint,
        timepoint_key=args.timepoint_key,
        seed=args.seed,
        coords=args.coords,
        apply_lognorm=not args.no_apply_lognorm,
        ct_real_reference=args.ct_real_reference,
        ct_real_n=args.ct_real_n,
        flat_pairing=args.flat_pairing,
        recon_real_reference=args.recon_real_reference,
        recon_real_n=args.recon_real_n,
    )

    skipped = res.pop("_skipped")
    notes = res.pop("_notes", [])
    tshape = res.pop("_target_shape", None)
    gshape = res.pop("_generated_shape", None)
    rows = sorted(res.items())
    if tshape is not None:
        print(f"target: {tshape[0]} cells, {tshape[1]} features")
    if gshape is not None:
        if len(gshape) == 3:
            print(f"generated: {gshape[0]} niches x {gshape[1]} points")
        else:
            print(f"generated: {gshape[0]} cells, {gshape[1]} feats (flat slide)")
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
