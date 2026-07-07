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
from paired_slides_eval.metrics._common import (
    build_paired_niches_from_flat,
    build_paired_niches_from_flat_fixed_centroids,
)
from paired_slides_eval.metrics.c2st import c2st_metrics
from paired_slides_eval.metrics.c2st_nn import c2st_nn_metrics
from paired_slides_eval.metrics.classifier_gap import classifier_accuracy_gap
from paired_slides_eval.metrics.concordance import _resolve_n_neighbors, cell_type_concordance
from paired_slides_eval.metrics.distances import point_to_shape, regression_metrics, shape_to_point
from paired_slides_eval.metrics.distribution import distribution_distance
from paired_slides_eval.metrics.expr_recon import expr_recon_gap, fixed_reference_mse
from paired_slides_eval.metrics.morans import morans_compare

ALL_GROUPS = (
    "regression",
    "psd",
    "spd",
    "distribution",
    "c2st",
    "c2st_nn",
    "moran",
    "concordance",
    "ct_gap",
    "recon",
)


def evaluate(
    target: TargetSlide,
    generated: GeneratedNiches | GeneratedSlide,
    *,
    classifier=None,
    regressor=None,
    classifier_spatial: bool = True,
    classifier_n_neighbors: int | None = None,
    auto_niche_from_flat: bool = True,
    auto_niche_max: int | None = None,
    flat_pairing: str = "fixed_target_ot",
    groups: tuple[str, ...] = ALL_GROUPS,
    prefix: str = "test",
    seed: int = 0,
    c2st_max_n: int = 2000,
    c2st_n_folds: int = 5,
    c2st_n_perm: int = 0,
    c2st_nn_k: int = 10,
    mmd_max_n: int = 2000,
    ot_max_n: int = 4000,
    moran_n_neighs: int = 6,
    ct_real_reference: str = "paired",
    ct_real_n: int = 2000,
    recon_real_reference: str = "fixed",
    recon_real_n: int = 2000,
) -> dict:
    """Compute every applicable metric for ``generated`` vs. ``target`` and return a flat dict.

    Groups: ``regression`` (needs ``generated.gt_*``), ``psd``/``spd``, ``distribution`` (MMD/EMD),
    ``c2st`` (per-cell joint + pos-only), ``c2st_nn`` (forgiving nearest-neighbour two-sample test
    pooled over spatial niches), ``moran`` (Moran's I over **all** generated cells),
    ``concordance`` (classifier agreement on generated vs paired-real niches) and ``ct_gap``
    (classifier accuracy gap real-vs-generated). The two classifier groups need a ``classifier``
    and the paired real niches ``generated.gt_*`` (``ct_gap`` also needs ``generated.gt_ct``).
    Pass a subset via ``groups`` to run only some.

    ``generated`` may be a :class:`~paired_slides_eval.contract.GeneratedNiches` (niche-shaped) or
    a flat :class:`~paired_slides_eval.contract.GeneratedSlide`. The label-free groups run on
    either. The classifier groups (``concordance``, ``ct_gap``) need paired niches: a
    ``GeneratedNiches`` supplies them directly, while for a flat slide they are reconstructed from
    geometry when ``auto_niche_from_flat`` is set. By default, ``flat_pairing="fixed_target_ot"``
    holds the target's fixed evaluation centroids (from ``subsampled_timepoint_idx``) constant and
    assigns each one to a unique generated cell by optimal transport in coordinate space; this
    mirrors NicheFlow's target-centroid pairing. If the target has no fixed centroids, evaluation
    falls back to ``"nearest_real"``: generated centroids paired to nearest real cells.
    Set ``auto_niche_max`` to cap how many fixed target centroids (or fallback generated centroids)
    are used. ``regression`` needs cell-for-cell matched ground truth, which a flat slide cannot
    provide, so it stays skipped for a flat slide regardless.

    ``ct_real_reference`` controls how ``ct/acc_real`` (in the ``ct_gap`` group) is measured:
    ``"paired"`` (default) scores the real niches paired to the generated centroids — model-dependent;
    ``"fixed"`` scores a seeded, model-independent sample of ``ct_real_n`` real target centroids, so
    ``acc_real`` is one constant across models and only ``acc_gen``/``acc_gap`` vary (use this for
    cross-model comparison tables). ``"fixed"`` needs the target's ``ct`` labels.

    ``recon_real_reference`` similarly controls ``recon/mse_real``; it defaults to ``"fixed"`` so the
    reconstruction gap uses a model-independent real-target baseline.
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

    if "c2st_nn" in groups:
        # Forgiving spatial two-sample test: per generated niche, the fraction of its spatial
        # neighbours whose joint-pool expression NN is a real cell. ~0.5 = indistinguishable.
        out.update(
            c2st_nn_metrics(
                target.x,
                target.pos,
                generated.flat_x,
                generated.flat_pos,
                prefix=prefix,
                max_n=c2st_max_n,
                spatial_k=c2st_nn_k,
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
    niche_model = classifier if classifier is not None else regressor
    if niche_model is not None and {"concordance", "ct_gap", "recon"}.intersection(groups):
        if not _has_paired_niches(generated) and isinstance(generated, GeneratedSlide):
            if auto_niche_from_flat:
                niche_gen, pair_note = _auto_paired_niches(
                    generated,
                    target,
                    niche_model,
                    spatial=classifier_spatial,
                    n_neighbors=classifier_n_neighbors,
                    max_centroids=auto_niche_max,
                    seed=seed,
                    flat_pairing=flat_pairing,
                )
                if niche_gen is not None:
                    notes.append(pair_note)

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
            # Optionally replace the (model-dependent, paired) acc_real with a fixed, seeded sample of
            # real target niches so acc_real is one constant across models — only acc_gen then varies.
            if ct_real_reference == "fixed" and target.ct is not None:
                from paired_slides_eval.metrics.classifier_gap import fixed_reference_accuracy

                p = f"{prefix}/" if prefix else ""
                acc_real = fixed_reference_accuracy(
                    target.x,
                    target.pos,
                    target.ct,
                    classifier,
                    spatial=classifier_spatial,
                    n_neighbors=classifier_n_neighbors,
                    n_centroids=ct_real_n,
                    seed=seed,
                )
                out[f"{p}ct/acc_real"] = acc_real
                out[f"{p}ct/acc_gap"] = abs(acc_real - out[f"{p}ct/acc_gen"])
                notes.append(
                    f"ct/acc_real from a fixed seeded sample of {ct_real_n} real target centroids "
                    "(model-independent), not the generated-centroid pairing"
                )
        else:
            skipped.append(
                "ct_gap (needs `classifier`, paired niches `gt_x/gt_pos` and true centroid labels "
                "`gt_ct` — for a flat slide, pass a target with `ct_key` so labels are available)"
            )

    if "recon" in groups:
        if regressor is not None and _has_paired_niches(niche_gen):
            out.update(
                expr_recon_gap(
                    niche_gen.x,
                    niche_gen.pos,
                    niche_gen.gt_x,
                    niche_gen.gt_pos,
                    regressor,
                    prefix=prefix,
                    n_neighbors=classifier_n_neighbors,
                )
            )
            # Optionally replace the (model-dependent, paired) mse_real with a fixed, seeded sample of
            # real target niches so mse_real is one constant across models — only mse_gen then varies.
            if recon_real_reference == "fixed":
                p = f"{prefix}/" if prefix else ""
                mse_real = fixed_reference_mse(
                    target.x,
                    target.pos,
                    regressor,
                    n_neighbors=classifier_n_neighbors,
                    n_centroids=recon_real_n,
                    seed=seed,
                )
                out[f"{p}recon/mse_real"] = mse_real
                out[f"{p}recon/mse_gap"] = abs(out[f"{p}recon/mse_gen"] - mse_real)
                notes.append(
                    f"recon/mse_real from a fixed seeded sample of {recon_real_n} real target "
                    "centroids (model-independent), not the generated-centroid pairing"
                )
        else:
            skipped.append(
                "recon (needs `regressor` and paired real niches — supply a `GeneratedNiches` with "
                "`gt_x/gt_pos`, or a flat slide with coords so they can be auto-built)"
            )

    out["_skipped"] = skipped
    out["_notes"] = notes
    return out


def _standardize_generated_coords(generated, coord_transform):
    """Map a generated slide/niches' coordinates into the target's standardised frame.

    Puts the generated coords in the same per-slide standardised frame the niche models and the
    classifier live in, so ``psd``/``spd``/``moran``/``c2st`` and the niche pairing are on a common
    scale. No-op when ``coord_transform`` is ``None``. Only the *generated* coordinates are touched —
    paired ``gt_pos`` already comes from the target (already standardised).
    """
    if coord_transform is None:
        return generated
    if isinstance(generated, GeneratedSlide):
        return GeneratedSlide(x=generated.x, pos=coord_transform.transform(generated.pos))
    b, n, _ = generated.pos.shape
    pos = coord_transform.transform(generated.pos.reshape(-1, generated.pos.shape[-1]))
    return GeneratedNiches(
        x=generated.x, pos=pos.reshape(b, n, -1), gt_x=generated.gt_x,
        gt_pos=generated.gt_pos, gt_ct=generated.gt_ct,
    )


def _detect_coord_space(gen_pos: np.ndarray, coord_transform) -> str:
    """Detect whether ``gen_pos`` is in the target's RAW frame (-> standardise) or already standardised.

    Standardised coords have per-axis std ~1; raw coords have std ~ the target's raw coord std (stored
    on ``coord_transform``). Picks whichever the generated per-axis std is closer to (log-ratio), so it
    is robust whether the raw std is large (the usual case) or itself near 1 (then either choice is a
    no-op anyway). Returns ``"standardize"`` or ``"passthrough"``.
    """
    eps = 1e-8
    gen_std = np.asarray(gen_pos, dtype=np.float64).std(axis=0) + eps
    raw_std = np.asarray(coord_transform.std, dtype=np.float64) + eps
    to_standardised = np.abs(np.log(gen_std)).mean()       # distance to std == 1
    to_raw = np.abs(np.log(gen_std / raw_std)).mean()      # distance to the target's raw std
    return "passthrough" if to_standardised <= to_raw else "standardize"


def _reconcile_generated(generated, target, *, coords: str = "auto"):
    """Bring ``generated`` coordinates into the target's frame; return ``(generated, notes)``.

    Expression is already reconciled by the caller via ``.project(target.pca)`` (gene-space ->
    projected, already-reduced -> passthrough). This handles the *coordinate* half centrally so every
    metric sees a consistent frame:

    * ``"auto"`` (default) — if the target carries a standardised coord frame
      (``target.coord_transform``, set for shared-PCA pickles), detect whether the generated coords are
      raw or already standardised and reconcile accordingly, recording the decision in the notes. This
      removes the old silent-mismatch footgun (forgetting to standardise OT-CFM coords).
    * ``"standardize"`` / ``"passthrough"`` — force the choice.
    """
    if coords not in ("auto", "standardize", "passthrough"):
        raise ValueError(f"coords must be auto|standardize|passthrough, got {coords!r}")
    notes: list[str] = []
    ct = target.coord_transform
    if ct is None:
        if coords == "standardize":
            raise ValueError(
                "coords='standardize' needs a shared-PCA .pkl target (it carries the coord frame); "
                "the given target has none."
            )
        return generated, notes

    decision = coords
    if coords == "auto":
        decision = _detect_coord_space(generated.flat_pos, ct)
        gs = np.round(np.asarray(generated.flat_pos).std(axis=0), 2).tolist()
        rs = np.round(np.asarray(ct.std), 2).tolist()
        notes.append(
            f"coords auto -> {decision} (generated per-axis std {gs} vs target raw std {rs}; "
            "standardised coords have std ~1)"
        )
    if decision == "standardize":
        generated = _standardize_generated_coords(generated, ct)
    return generated, notes


def evaluate_files(
    target,
    generated,
    *,
    ct_key: str | None = None,
    classifier=None,
    regressor=None,
    classifier_kwargs: dict | None = None,
    regressor_kwargs: dict | None = None,
    groups: tuple[str, ...] = ALL_GROUPS,
    expr_key: str | None = None,
    spatial_key: str = "spatial",
    timepoint: str | None = None,
    timepoint_key: str | None = None,
    seed: int = 0,
    coords: str = "auto",
    apply_lognorm: bool = True,
    **evaluate_kwargs,
) -> dict:
    """Evaluate generated cells against a target slide straight from files — the one-call front door.

    The single entry point: it loads both sides, reconciles them into one space, and runs
    :func:`evaluate`. The CLI (``python -m paired_slides_eval.evaluate``) is a thin wrapper over this.

    Space reconciliation is automatic, so the *same* call works for every model:

    * **Expression** — for a ``.pkl`` target carrying the shared-PCA recipe, gene-space generated cells
      (e.g. OT-CFM) are projected into the whitened shared basis and already-reduced cells (NicheFlow)
      pass through (dimension auto-detect in :meth:`~paired_slides_eval.contract.GeneratedSlide.project`).
    * **Coordinates** — ``coords="auto"`` detects whether the generated coords are raw or already
      standardised and maps them into the target's frame, logging the decision (no silent mismatch).

    Args:
        target: the target slide — a preprocessed-slide ``.pkl`` (the recommended, cross-model
            comparable path: whitened shared PCA + standardised coords). Raise an error when a ``.h5ad``/
            ``AnnData``is loaded since it's not comparable across models.
        generated: the generated cells as a path — ``.h5ad`` (flat or niche-shaped), ``.npz``, or
            ``.pkl``.
        ct_key: ``obs`` column with cell types (``.h5ad`` target only; a ``.pkl`` already has labels).
        classifier: a ready classifier module, or a path to a ``.ckpt`` (enables the ``ct/*`` groups).
        regressor: a ready masked-centroid expression regressor module, or a path to a ``.ckpt``
            (enables the ``recon`` group).
        classifier_kwargs: net hyperparameters for ``build_spatial_classifier`` when ``classifier`` is
            a path (``hidden_dim`` / ``num_heads`` / ``mask_centroid``).
        regressor_kwargs: net hyperparameters for ``build_spatial_regressor`` when ``regressor`` is
            a path.
        coords: ``"auto"`` (default) / ``"standardize"`` / ``"passthrough"`` — how generated coords are
            reconciled to the target frame (see :func:`_reconcile_generated`).
        apply_lognorm: forwarded to the shared-PCA recipe — ``False`` if the generated gene-space cells
            are already log-normalised (see ``docs/comparability_plan.md``).
        groups / seed / **evaluate_kwargs: forwarded to :func:`evaluate` (e.g.
            ``ct_real_reference="fixed"`` for a cross-model-comparable ``ct/acc_real``).

    Returns the flat ``{prefix/group/metric: value}`` dict (plus ``_skipped`` / ``_notes`` and the
    private ``_target_shape`` / ``_generated_shape`` used by the CLI to print a header).
    """
    notes: list[str] = []
    if isinstance(target, str) and target.endswith(".pkl"):
        target_slide = TargetSlide.from_dataclass(
            target, timepoint=timepoint, apply_lognorm=apply_lognorm
        )
    else:
        raise ValueError(
            "target loaded from AnnData with a per-target PCA — NOT cross-model comparable; pass a "
            "shared preprocess_pair .pkl as the target to compare models."
        )

    gen = _load_generated(generated)
    gen = gen.project(target_slide.pca)
    gen, coord_notes = _reconcile_generated(gen, target_slide, coords=coords)

    clf = classifier
    if isinstance(classifier, str):
        clf = build_spatial_classifier(
            classifier, target_slide.x.shape[1], target_slide.n_classes,
            **(classifier_kwargs or {}),
        )

    reg = regressor
    if isinstance(regressor, str):
        reg = build_spatial_regressor(
            regressor,
            target_slide.x.shape[1],
            **(regressor_kwargs or {}),
        )

    res = evaluate(
        target_slide,
        gen,
        classifier=clf,
        regressor=reg,
        groups=groups,
        seed=seed,
        **evaluate_kwargs,
    )
    res["_notes"] = notes + coord_notes + res.get("_notes", [])
    res["_target_shape"] = tuple(target_slide.x.shape)
    res["_generated_shape"] = tuple(gen.x.shape)
    return res


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
    flat_pairing: str,
) -> tuple[GeneratedNiches | None, str]:
    """Reconstruct paired niches from a flat slide via geometry, for the classifier groups.

    The niche size matches what the spatial classifier was trained on (``_resolve_n_neighbors``);
    a gene-only classifier only reads the centroid, so a single point suffices. Returns ``None``
    when there are too few cells to form a niche. ``gt_ct`` is filled only when the target carries
    cell-type labels (``target.ct``), so ``ct_gap`` is enabled exactly when those labels exist.
    """
    if flat_pairing not in ("fixed_target_ot", "nearest_real"):
        raise ValueError(
            f"flat_pairing must be 'fixed_target_ot' or 'nearest_real', got {flat_pairing!r}"
        )

    gen_pos = generated.flat_pos
    if len(gen_pos) < 1 or len(target.pos) < 1:
        return None, "classifier niches not auto-built (empty generated or target slide)"

    k = _resolve_n_neighbors(n_neighbors, classifier) if spatial else 1
    rng = np.random.default_rng(seed)

    if flat_pairing == "fixed_target_ot" and target.eval_centroid_indices is not None:
        target_centroids = np.asarray(target.eval_centroid_indices, dtype=np.int64)
        if max_centroids is not None and len(target_centroids) > max_centroids:
            target_centroids = rng.choice(target_centroids, max_centroids, replace=False)
        nx, npos, gt_x, gt_pos, gt_ct = build_paired_niches_from_flat_fixed_centroids(
            generated.flat_x,
            gen_pos,
            target.x,
            target.pos,
            k,
            real_ct=target.ct,
            target_centroid_indices=target_centroids,
        )
        niche = GeneratedNiches(x=nx, pos=npos, gt_x=gt_x, gt_pos=gt_pos, gt_ct=gt_ct)
        return (
            niche,
            "classifier niches auto-built from the flat slide via fixed target centroids + OT "
            f"assignment ({niche.x.shape[0]} target centroids matched to generated cells)",
        )

    centroid_indices = None
    if max_centroids is not None and len(gen_pos) > max_centroids:
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
    niche = GeneratedNiches(x=nx, pos=npos, gt_x=gt_x, gt_pos=gt_pos, gt_ct=gt_ct)
    note = (
        "classifier niches auto-built from the flat slide via nearest-real fallback "
        f"({niche.x.shape[0]} generated centroids paired to nearest real cells)"
    )
    if flat_pairing == "fixed_target_ot":
        note += "; target has no fixed eval_centroid_indices"
    return niche, note


def build_spatial_classifier(
    ckpt_path: str,
    input_dim: int,
    output_dim: int,
    *,
    hidden_dim: int = 64,
    num_heads: int = 4,
    mask_centroid: bool = True,
):
    """Reconstruct the spatial SetTransformer classifier and load a checkpoint into it.

    The net is **expression-only** (no coordinates); hyperparameters must match what was trained, and
    ``load_spatial_classifier`` attaches the training KNN ``k`` so the classifier metrics build
    identically sized niches.
    """
    import torch

    from paired_slides_eval.classifier.nets import SpatialCTClassifierNet
    from paired_slides_eval.metrics._common import load_spatial_classifier

    clf = SpatialCTClassifierNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mask_centroid=mask_centroid,
    )
    # weights_only=False: a Lightning .ckpt carries more than tensors (e.g. the OmegaConf
    # hyper_parameters), which the torch>=2.6 weights-only default refuses. The classifier
    # checkpoint is a trusted local file produced by this package's trainer.
    load_spatial_classifier(clf, torch.load(ckpt_path, map_location="cpu", weights_only=False))
    return clf


def build_spatial_regressor(
    ckpt_path: str,
    input_dim: int,
    *,
    hidden_dim: int = 64,
    num_heads: int = 4,
    mask_centroid: bool = True,
):
    """Reconstruct the masked-centroid expression regressor and load a checkpoint into it.

    The regressor reuses ``SpatialCTClassifierNet`` with ``output_dim == input_dim``; checkpoint
    loading also attaches the training KNN ``k`` so reconstruction eval rebuilds matching niches.
    """
    return build_spatial_classifier(
        ckpt_path,
        input_dim,
        input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mask_centroid=mask_centroid,
    )


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
        help="the target slide: a preprocessed-slide .pkl (scored in the shared whitened X_pca; "
        "--expr_key ignored)",
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
    # Spatial classifier net hyperparameters (must match training; only used with --classifier).
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


# Usage: python -m paired_slides_eval.evaluate --target target.h5ad --generated generated.npz
if __name__ == "__main__":
    _main()
