"""Top-level entry point: run the whole evaluation suite on one (target, generated) pair.

``evaluate`` returns a flat ``{prefix/group/metric: value}`` dict, convenient to write straight to
a results CSV. Each metric group is optional and is skipped (with a note in the returned
``_skipped`` list) when its inputs are absent — e.g. the classifier groups need a classifier +
paired niches, regression needs matched ground truth.

This is the **standalone** path: it takes a real ``TargetSlide`` and the model's
``GeneratedNiches`` (both built from AnnData) and never touches the flow model. To go all the way
from a checkpoint + raw slides, see :func:`paired_slides_eval.pipeline.run.run_pipeline`.
"""

from __future__ import annotations

import numpy as np

from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide, TargetSlide
from paired_slides_eval.data.anndata import read_anndata
from paired_slides_eval.metrics._common import build_paired_niches_from_flat
from paired_slides_eval.metrics.c2st import c2st_metrics
from paired_slides_eval.metrics.classifier_gap import classifier_accuracy_gap
from paired_slides_eval.metrics.concordance import _resolve_n_neighbors, cell_type_concordance
from paired_slides_eval.metrics.distances import point_to_shape, regression_metrics, shape_to_point
from paired_slides_eval.metrics.distribution import distribution_distance
from paired_slides_eval.metrics.morans import morans_compare

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
    auto_niche_from_flat: bool = True,
    auto_niche_max: int | None = None,
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

    ``generated`` may be a :class:`~paired_slides_eval.contract.GeneratedNiches` (niche-shaped) or
    a flat :class:`~paired_slides_eval.contract.GeneratedSlide`. The label-free groups run on
    either. The classifier groups (``concordance``, ``ct_gap``) need paired niches: a
    ``GeneratedNiches`` supplies them directly, while for a flat slide they are reconstructed from
    geometry when ``auto_niche_from_flat`` is set (each generated cell's neighbourhood paired to the
    nearest real cell's; ``ct_gap`` additionally needs the target's ``ct`` labels). Set
    ``auto_niche_max`` to cap how many generated cells are used as niche centroids. ``regression``
    needs cell-for-cell matched ground truth, which a flat slide cannot provide, so it stays skipped
    for a flat slide regardless.
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

    # The classifier groups compare each generated niche to a *paired real* niche. A niche-aware
    # model supplies that pairing on the `GeneratedNiches` (`gt_x`/`gt_pos`[/`gt_ct`]); for a flat
    # whole-slide model we reconstruct it from geometry so the metrics still run. Build it once and
    # share between concordance and ct_gap.
    notes: list[str] = []
    niche_gen = generated
    if classifier is not None and ("concordance" in groups or "ct_gap" in groups):
        if not _has_paired_niches(generated) and isinstance(generated, GeneratedSlide):
            if auto_niche_from_flat:
                niche_gen = _auto_paired_niches(
                    generated,
                    target,
                    classifier,
                    spatial=classifier_spatial,
                    n_neighbors=classifier_n_neighbors,
                    max_centroids=auto_niche_max,
                    seed=seed,
                )
                if niche_gen is not None:
                    notes.append(
                        f"classifier niches auto-built from the flat slide via geometry "
                        f"({niche_gen.x.shape[0]} centroids paired to nearest real cells)"
                    )

    if "concordance" in groups:
        if classifier is not None and _has_paired_niches(niche_gen):
            out.update(
                cell_type_concordance(
                    niche_gen.x,
                    niche_gen.pos,
                    niche_gen.gt_x,
                    niche_gen.gt_pos,
                    classifier,
                    prefix=prefix,
                    spatial=classifier_spatial,
                    n_neighbors=classifier_n_neighbors,
                    n_classes=target.n_classes,
                )
            )
        else:
            skipped.append(
                "concordance (needs `classifier` and paired real niches — supply a "
                "`GeneratedNiches` with `gt_x/gt_pos`, or a flat slide with coords so they can be "
                "auto-built)"
            )

    if "ct_gap" in groups:
        if (
            classifier is not None
            and _has_paired_niches(niche_gen)
            and getattr(niche_gen, "gt_ct", None) is not None
        ):
            out.update(
                classifier_accuracy_gap(
                    niche_gen.x,
                    niche_gen.pos,
                    niche_gen.gt_x,
                    niche_gen.gt_pos,
                    niche_gen.gt_ct,
                    classifier,
                    prefix=prefix,
                    spatial=classifier_spatial,
                    n_neighbors=classifier_n_neighbors,
                )
            )
        else:
            skipped.append(
                "ct_gap (needs `classifier`, paired niches `gt_x/gt_pos` and true centroid labels "
                "`gt_ct` — for a flat slide, pass a target with `ct_key` so labels are available)"
            )

    out["_skipped"] = skipped
    out["_notes"] = notes
    return out


def evaluate_files(
    target,
    generated,
    *,
    ct_key: str | None = None,
    n_pcs: int | None = None,
    classifier=None,
    groups: tuple[str, ...] = ALL_GROUPS,
    expr_key: str | None = None,
    spatial_key: str = "spatial",
    seed: int = 0,
    **evaluate_kwargs,
) -> dict:
    """Evaluate generated cells against a target slide straight from files — the one-call front door.

    This is the headline entry point for the common case: you generated cells with your own model
    (in your own repo) and just want the metrics. It loads both sides, puts them in a shared feature
    space, and runs :func:`evaluate` — no dataclasses to assemble by hand.

    Args:
        target: the target slide — a ``.h5ad`` (raw genes + ``obsm['spatial']``) or an ``AnnData``;
            a ``.pkl`` is also accepted (a preprocessed-slide pickle, via
            :meth:`~paired_slides_eval.contract.TargetSlide.from_dataclass`).
        generated: the generated cells as a path — ``.h5ad`` (flat ``X``+``obsm['spatial']`` or
            niche-shaped with ``obs['niche_id']``), ``.npz``, or ``.pkl``.
        ct_key: ``obs`` column with cell types on the target (needed for the ``ct/*`` metrics).
        n_pcs: fit a PCA on the target to ``n_pcs`` and project both sides into it so they share a
            basis. Leave ``None`` only if target and generated are already in the same space.
        classifier: a ready classifier module, or a path to a ``.ckpt`` (enables the ``ct/*``
            metrics); ``None`` skips them.
        groups / seed / **evaluate_kwargs: forwarded to :func:`evaluate`.

    Returns the flat ``{prefix/group/metric: value}`` dict (plus ``_skipped`` / ``_notes``).
    """
    if isinstance(target, str) and target.endswith(".pkl"):
        target_slide = TargetSlide.from_dataclass(target)
    else:
        target_slide = TargetSlide.from_anndata(
            target, ct_key=ct_key, expr_key=expr_key, spatial_key=spatial_key, n_pcs=n_pcs
        )

    gen = _load_generated(generated).project(target_slide.pca)

    clf = classifier
    if isinstance(classifier, str):
        clf = build_spatial_classifier(
            classifier, target_slide.x.shape[1], target_slide.n_classes
        )

    return evaluate(target_slide, gen, classifier=clf, groups=groups, seed=seed, **evaluate_kwargs)


def _has_paired_niches(generated) -> bool:
    """True if ``generated`` carries the paired real microenvironments the classifier groups need."""
    return (
        getattr(generated, "gt_x", None) is not None
        and getattr(generated, "gt_pos", None) is not None
    )


def _auto_paired_niches(
    generated: GeneratedSlide,
    target: TargetSlide,
    classifier,
    *,
    spatial: bool,
    n_neighbors: int | None,
    max_centroids: int | None,
    seed: int,
) -> GeneratedNiches | None:
    """Reconstruct paired niches from a flat slide via geometry, for the classifier groups.

    The niche size matches what the spatial classifier was trained on (``_resolve_n_neighbors``);
    a gene-only classifier only reads the centroid, so a single point suffices. Returns ``None``
    when there are too few cells to form a niche. ``gt_ct`` is filled only when the target carries
    cell-type labels (``target.ct``), so ``ct_gap`` is enabled exactly when those labels exist.
    """
    gen_pos = generated.flat_pos
    if len(gen_pos) < 1 or len(target.pos) < 1:
        return None

    k = _resolve_n_neighbors(n_neighbors, classifier) if spatial else 1

    centroid_indices = None
    if max_centroids is not None and len(gen_pos) > max_centroids:
        rng = np.random.default_rng(seed)
        centroid_indices = rng.choice(len(gen_pos), max_centroids, replace=False)

    nx, npos, gt_x, gt_pos, gt_ct = build_paired_niches_from_flat(
        generated.flat_x,
        gen_pos,
        target.x,
        target.pos,
        k,
        real_ct=target.ct,
        centroid_indices=centroid_indices,
    )
    return GeneratedNiches(x=nx, pos=npos, gt_x=gt_x, gt_pos=gt_pos, gt_ct=gt_ct)


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

    from paired_slides_eval.classifier.nets import SpatialCTClassifierNet
    from paired_slides_eval.metrics._common import load_spatial_classifier

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


def _generated_from_mapping(m) -> GeneratedNiches | GeneratedSlide:
    """Build generated cells from an ``x``/``pos`` mapping (``.npz`` or an unpickled dict).

    Niche-shaped if ``x`` is 3-D ``(B, N, D)`` (optionally with ``gt_x``/``gt_pos``/``gt_ct``),
    else a flat ``GeneratedSlide`` from 2-D ``x``/``pos``.
    """
    x = np.asarray(m["x"])
    if x.ndim == 3:
        extra = {k: np.asarray(m[k]) for k in ("gt_x", "gt_pos", "gt_ct") if k in m}
        return GeneratedNiches(x=x, pos=np.asarray(m["pos"]), **extra)
    return GeneratedSlide(x=x, pos=np.asarray(m["pos"]))


def _load_generated(path: str, *, niche_key: str = "niche_id") -> GeneratedNiches | GeneratedSlide:
    """Load generated cells, auto-detecting niche-shaped vs flat.

    ``.h5ad``: niche-shaped if ``obs[niche_key]`` is present, else a flat ``GeneratedSlide``.
    ``.npz``: niche-shaped if ``x`` is 3-D ``(B, N, D)`` (optionally with ``gt_x``/``gt_pos``/
    ``gt_ct``), else a flat ``GeneratedSlide`` from 2-D ``x``/``pos``.
    ``.pkl``: a generator result object (any object with a ``to_generated_niches`` method) or a
    dict with the same ``x``/``pos``[/``gt_*``] arrays as the ``.npz`` form.
    """
    if str(path).endswith(".h5ad"):
        adata = read_anndata(path)
        if niche_key in adata.obs:
            return GeneratedNiches.from_anndata(adata, niche_key=niche_key)
        return GeneratedSlide.from_anndata(adata)

    if str(path).endswith(".pkl"):
        import pickle

        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if hasattr(obj, "to_generated_niches"):  # a generator result object
            return obj.to_generated_niches()
        if isinstance(obj, (GeneratedNiches, GeneratedSlide)):
            return obj
        if isinstance(obj, dict) and "x" in obj and "pos" in obj:
            return _generated_from_mapping(obj)
        raise ValueError(
            f"Unrecognised generated .pkl contents ({type(obj).__name__}). Expected a "
            "GenerationResult, a GeneratedNiches/GeneratedSlide, or a dict with x/pos arrays. "
            "A preprocessed-slide pickle is a *real* slide — load it as a target with "
            "TargetSlide.from_dataclass instead."
        )

    return _generated_from_mapping(np.load(path))


def _main() -> None:
    import argparse
    import csv
    import os

    ap = argparse.ArgumentParser(
        description="Run the metric suite on a (target slide, generated cells) pair."
    )
    ap.add_argument(
        "--target",
        required=True,
        help="the target slide: a .h5ad (raw genes + coords) or a preprocessed-slide .pkl "
        "(X_pca already reduced; --n_pcs/--expr_key are then ignored)",
    )
    ap.add_argument(
        "--generated",
        required=True,
        help="generated cells. Niche-shaped: .npz with x (B,N,P), pos (B,N,P) "
        "[+ gt_x/gt_pos/gt_ct], an .h5ad with obs['niche_id'], or a .pkl (GenerationResult / dict "
        "of arrays). Flat (whole-slide): .npz with 2-D x/pos, or an .h5ad with X + "
        "obsm['spatial'] (niche metrics are then auto-built from geometry when a classifier is "
        "given).",
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

    if str(args.target).endswith(".pkl"):
        target = TargetSlide.from_dataclass(args.target, timepoint=args.timepoint)
    else:
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
    notes = res.pop("_notes", [])
    rows = sorted(res.items())
    print(f"target: {target.x.shape[0]} cells, {target.x.shape[1]} features")
    if generated.x.ndim == 3:
        print(f"generated: {generated.x.shape[0]} niches x {generated.x.shape[1]} points")
    else:
        print(f"generated: {generated.x.shape[0]} cells, {generated.x.shape[1]} feats (flat slide)")
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


# Usage: python -m paired_slides_eval.evaluate --target target.h5ad --generated generated.npz
if __name__ == "__main__":
    _main()
