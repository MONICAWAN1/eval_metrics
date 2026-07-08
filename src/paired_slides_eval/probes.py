"""Classifier / regressor probes and paired-niche assembly for the ``ct/*`` and
``recon`` groups.

Loads a trained spatial classifier/regressor from a checkpoint, and — for a flat
whole-slide model that supplies no niche pairing — reconstructs paired
real/generated niches from geometry so the classifier metrics can still run.

"""

from __future__ import annotations

import numpy as np

from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide, TargetSlide
from paired_slides_eval.metrics._common import (
    build_paired_niches_from_flat,
    build_paired_niches_from_flat_fixed_centroids,
)
from paired_slides_eval.metrics.concordance import _resolve_n_neighbors


def _has_paired_niches(generated) -> bool:
    """True if ``generated`` carries the paired real microenvironments the
    classifier groups need."""
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
    """Reconstruct paired niches from a flat slide via geometry, for the
    classifier groups.

    The niche size matches what the spatial classifier was trained on (``_resolve_n_neighbors``);
    a gene-only classifier only reads the centroid, so a single point suffices. Returns ``None``
    when there are too few cells to form a niche. ``gt_ct`` is filled only when the target carries
    cell-type labels (``target.ct``), so ``ct_gap`` is enabled exactly when those labels exist.

    """
    if flat_pairing not in ("fixed_target_ot", "nearest_real"):
        raise ValueError(
            f"flat_pairing must be 'fixed_target_ot' or 'nearest_real', got {flat_pairing!r}",
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
    """Reconstruct the spatial SetTransformer classifier and load a checkpoint
    into it.

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
    """Reconstruct the masked-centroid expression regressor and load a
    checkpoint into it.

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
