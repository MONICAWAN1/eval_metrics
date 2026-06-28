"""Cell-type classifier accuracy gap (``ct/acc_real``, ``ct/acc_gen``, ``ct/acc_gap``).

A complementary read on the concordance probe (see :mod:`paired_slides_eval.metrics.concordance`).
The intuition: run the same trained cell-type classifier on the real target niches and on the
generated niches, scoring each against the true centroid labels. If the generated slide is
realistic, the classifier should be about as accurate on it as on the real slide — so a small
``|acc_real - acc_gen|`` gap is good (the generated niches are as "classifiable" as the real ones),
and a large gap flags that generation distorts the local structure the classifier relies on.

This reuses the niche assembly (``build_microenv_points``) and the ``n_neighbors`` resolution from
the concordance metric so both probes feed the classifier identically sized microenvironments.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

from paired_slides_eval.metrics._common import (
    build_microenv_points,
    build_paired_niches_from_flat,
)
from paired_slides_eval.metrics.concordance import _resolve_n_neighbors


def _classify_niches(classifier, feats, device):
    """Run a frozen classifier over assembled niche features and return argmax labels."""
    import torch

    with torch.no_grad():
        return (
            classifier(torch.as_tensor(feats, dtype=torch.float32, device=device))
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )


def fixed_reference_accuracy(
    real_x: np.ndarray,
    real_pos: np.ndarray,
    real_ct: np.ndarray,
    classifier,
    *,
    spatial: bool = True,
    n_neighbors: int | None = None,
    n_centroids: int = 2000,
    seed: int = 0,
) -> float:
    """Classifier accuracy on a **model-independent** sample of real target niches.

    The paired ``ct/acc_real`` ([`classifier_accuracy_gap`]) is measured on whichever real niches got
    paired to a model's generated centroids, so it drifts between models (and between the flat-slide
    geometric pairing and a niche model's own grid). This instead samples a fixed, seeded set of
    target cells as centroids and builds their real microenvironments directly — making ``acc_real`` a
    constant property of (classifier, target, niche size), comparable across every model. Used by
    :func:`~paired_slides_eval.evaluate.evaluate` when ``ct_real_reference='fixed'``.
    """
    import torch

    real_pos = np.asarray(real_pos)
    n_real = len(real_pos)
    rng = np.random.default_rng(seed)
    centroids = (
        rng.choice(n_real, n_centroids, replace=False) if n_real > n_centroids else np.arange(n_real)
    )

    k = _resolve_n_neighbors(n_neighbors, classifier) if spatial else 1
    # gen == real here, so the "generated" niche assembled at each sampled centroid IS the real niche.
    niche_x, niche_pos, _, _, gt_ct = build_paired_niches_from_flat(
        real_x, real_pos, real_x, real_pos, k, real_ct=real_ct, centroid_indices=centroids
    )

    feats = build_microenv_points(niche_x, niche_pos, k)[0] if spatial else niche_x[:, 0, :]
    classifier.eval()
    device = next(classifier.parameters()).device
    y_pred = _classify_niches(classifier, feats, device)
    return float(accuracy_score(np.asarray(gt_ct, dtype=np.int64), y_pred))


def classifier_accuracy_gap(
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    gt_x: np.ndarray,
    gt_pos: np.ndarray,
    gt_ct: np.ndarray,
    classifier,
    *,
    prefix: str = "",
    spatial: bool = True,
    n_neighbors: int | None = None,
) -> dict[str, float]:
    """Accuracy of ``classifier`` on real vs. generated niches against true labels, and the gap.

    Args:
        gen_x / gen_pos: generated microenvironments ``(B, N, n_pcs)`` / ``(B, N, coord)``,
            centroid at point 0.
        gt_x / gt_pos: the paired real target microenvironments (same centroids), same shapes.
        gt_ct: ``(B,)`` true cell-type label of each paired real centroid.
        classifier: a frozen ``torch.nn.Module`` (the trained cell-type classifier). Spatial nets
            take the ``[expression | relative_position]`` point set ``(B, k, n_pcs + coord)``; the
            gene-only net takes the centroid expression ``(B, n_pcs)``.
        spatial: whether ``classifier`` is a spatial SetTransformer-based net or gene-only.
        n_neighbors: microenvironment size for the spatial classifier; ``None`` -> the value
            recorded on the classifier at training time (matching the concordance probe).

    Returns:
        ``{prefix/ct/acc_real, prefix/ct/acc_gen, prefix/ct/acc_gap}`` — accuracy on the real
        niches, on the generated niches (both vs. the true centroid labels), and ``|real - gen|``.
    """
    import torch

    p = f"{prefix}/" if prefix else ""

    if spatial:
        k = _resolve_n_neighbors(n_neighbors, classifier)
        feats_gen, _ = build_microenv_points(gen_x, gen_pos, k)
        feats_real, _ = build_microenv_points(gt_x, gt_pos, k)
    else:
        feats_gen, feats_real = gen_x[:, 0, :], gt_x[:, 0, :]

    classifier.eval()
    device = next(classifier.parameters()).device
    with torch.no_grad():
        y_gen = (
            classifier(torch.as_tensor(feats_gen, dtype=torch.float32, device=device))
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )
        y_real = (
            classifier(torch.as_tensor(feats_real, dtype=torch.float32, device=device))
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )

    y_true = np.asarray(gt_ct, dtype=np.int64)
    acc_real = float(accuracy_score(y_true, y_real))
    acc_gen = float(accuracy_score(y_true, y_gen))
    return {
        f"{p}ct/acc_real": acc_real,
        f"{p}ct/acc_gen": acc_gen,
        f"{p}ct/acc_gap": abs(acc_real - acc_gen),
    }
