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
from paired_slides_eval.loaders import _load_generated
from paired_slides_eval.metrics.c2st import c2st_metrics
from paired_slides_eval.metrics.c2st_nn import c2st_nn_metrics
from paired_slides_eval.metrics.classifier_gap import classifier_accuracy_gap
from paired_slides_eval.metrics.concordance import cell_type_concordance
from paired_slides_eval.metrics.distances import regression_metrics
from paired_slides_eval.metrics.distribution import distribution_distance
from paired_slides_eval.metrics.expr_recon import expr_recon_gap, fixed_reference_mse
from paired_slides_eval.metrics.morans import morans_compare
from paired_slides_eval.probes import (
    _auto_paired_niches,
    _has_paired_niches,
    build_spatial_classifier,
    build_spatial_regressor,
)
from paired_slides_eval.reconcile import _reconcile_generated

ALL_GROUPS = (
    "regression",
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
    """Compute every applicable metric for ``generated`` vs. ``target`` and
    return a flat dict.

    Groups: ``regression`` (needs ``generated.gt_*``), ``distribution`` (MMD/EMD), ``c2st``
    (per-cell joint + expression-only), ``c2st_nn`` (forgiving nearest-neighbour two-sample test
    pooled over spatial niches), ``moran`` (Moran's I over **all** generated cells),
    ``concordance`` (classifier agreement on generated vs paired-real niches), ``ct_gap``
    (classifier accuracy gap real-vs-generated), and ``recon`` (masked-centroid expression
    reconstruction). The two classifier groups need a ``classifier`` and the paired real niches
    ``generated.gt_*`` (``ct_gap`` also needs ``generated.gt_ct``). Pass a subset via ``groups`` to
    run only some.

    ``generated`` may be a :class:`~paired_slides_eval.contract.GeneratedNiches` (niche-shaped) or
    a flat :class:`~paired_slides_eval.contract.GeneratedSlide`. The label-free groups run on
    either. The classifier and reconstruction groups need paired niches: a ``GeneratedNiches``
    supplies them directly, while for a flat slide they are reconstructed from geometry when
    ``auto_niche_from_flat`` is set. By default, ``flat_pairing="fixed_target_ot"`` holds the
    target's fixed evaluation centroids (from ``subsampled_timepoint_idx``) constant and assigns each
    one to a unique generated cell by optimal transport in coordinate space. If the target has no
    fixed centroids, evaluation falls back to ``"nearest_real"``: generated centroids paired to
    nearest real cells. Set ``auto_niche_max`` to cap how many fixed target centroids (or fallback
    generated centroids) are used. ``regression`` needs cell-for-cell matched ground truth, which a
    flat slide cannot provide, so it stays skipped for a flat slide regardless.

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
                    generated.x,
                    generated.gt_x,
                    generated.pos,
                    generated.gt_pos,
                    prefix=prefix,
                ),
            )
        else:
            skipped.append(
                "regression (needs niche-shaped `GeneratedNiches` with matched `gt_x`/`gt_pos`)",
            )

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
            ),
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
            ),
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
            ),
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
            ),
        )

    # The classifier/reconstruction groups compare generated niches to paired real niches. A
    # niche-aware model supplies that pairing on the `GeneratedNiches` (`gt_x`/`gt_pos`[/`gt_ct`]);
    # for a flat whole-slide model we reconstruct it from geometry so the metrics still run.
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
                ),
            )
        else:
            skipped.append(
                "concordance (needs `classifier` and paired real niches — supply a "
                "`GeneratedNiches` with `gt_x/gt_pos`, or a flat slide with coords so they can be "
                "auto-built)",
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
                ),
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
                    "(model-independent), not the generated-centroid pairing",
                )
        else:
            skipped.append(
                "ct_gap (needs `classifier`, paired niches `gt_x/gt_pos` and true centroid labels "
                "`gt_ct` — for a flat slide, pass a target with `ct_key` so labels are available)",
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
                ),
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
                    "centroids (model-independent), not the generated-centroid pairing",
                )
        else:
            skipped.append(
                "recon (needs `regressor` and paired real niches — supply a `GeneratedNiches` with "
                "`gt_x/gt_pos`, or a flat slide with coords so they can be auto-built)",
            )

    out["_skipped"] = skipped
    out["_notes"] = notes
    return out


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
    """Evaluate generated cells against a target slide straight from files — the
    one-call front door.

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
            target,
            timepoint=timepoint,
            apply_lognorm=apply_lognorm,
        )
    else:
        raise ValueError(
            "target loaded from AnnData with a per-target PCA — NOT cross-model comparable; pass a "
            "shared preprocess_pair .pkl as the target to compare models.",
        )

    gen = _load_generated(generated)
    gen = gen.project(target_slide.pca)
    gen, coord_notes = _reconcile_generated(gen, target_slide, coords=coords)

    clf = classifier
    if isinstance(classifier, str):
        clf = build_spatial_classifier(
            classifier,
            target_slide.x.shape[1],
            target_slide.n_classes,
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


# `python -m paired_slides_eval.evaluate ...` stays working via the CLI in `cli.py`.
if __name__ == "__main__":
    from paired_slides_eval.cli import main

    main()
