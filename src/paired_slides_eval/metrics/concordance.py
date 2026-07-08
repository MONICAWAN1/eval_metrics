"""Cell-type classifier concordance (``ct/*``) — neutral held-out classifier.

A **neutral** cell-type classifier — trained on a held-out same-mouse slide that is *neither* the
source *nor* the target (the ``abca_aligned_clf.pkl`` slide, projected into the source+target PCA
basis + label space) — is applied to **both** the generated niche and its **paired real target
niche** (same centroid). Because the same annotator scores both sides and it never saw the target
slide, this avoids the leakage/circularity of the old nearest-real-cell pseudo-label.

We report, per niche:
- **agreement** of the classifier's labels on generated vs. real niches — weighted F1 / accuracy
  (``ct/f1``, ``ct/acc``): is each generated niche labelled like the real one it was meant to be?
- **composition divergence** of the two label-proportion histograms — KL, total variation, and
  Jensen-Shannon (``ct/prop_kl``, ``ct/prop_tv``, ``ct/prop_jsd``): does the generated cell-type
  mix match the real one under that annotator?

Needs a trained classifier and the **paired real target niches** (``GeneratedNiches.gt_x/gt_pos``)
— not the target's own annotations. See ``paired_slides_eval.classifier`` for training one.

"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from paired_slides_eval.metrics._common import (
    build_microenv_points,
    jensen_shannon,
    kl_divergence,
    proportions,
    total_variation,
)

# KNN k used only when neither the caller nor the loaded classifier supplies one (e.g. a checkpoint
# predating the recorded value). Matches the training config default.
_DEFAULT_N_NEIGHBORS = 10


def _resolve_n_neighbors(n_neighbors: int | None, classifier) -> int:
    """Pick the microenvironment size for the spatial classifier, matching
    training.

    Priority: an explicit ``n_neighbors`` overrides; otherwise use the value the classifier
    recorded at training time (attached by
    :func:`~paired_slides_eval.metrics._common.load_spatial_classifier`); otherwise fall back to
    :data:`_DEFAULT_N_NEIGHBORS` and warn, since a mismatch feeds the net a different-sized niche
    than it trained on.

    """
    if n_neighbors is not None:
        return int(n_neighbors)
    recorded = getattr(classifier, "n_neighbors", None)
    if recorded is not None:
        return int(recorded)
    warnings.warn(
        f"n_neighbors was not given and the classifier has no recorded training value; "
        f"falling back to {_DEFAULT_N_NEIGHBORS}. This may not match the niche size the net was "
        f"trained on. Load the classifier with load_spatial_classifier (so the trained value "
        f"travels with it) or pass n_neighbors explicitly.",
        stacklevel=2,
    )
    return _DEFAULT_N_NEIGHBORS


def cell_type_concordance(
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    gt_x: np.ndarray,
    gt_pos: np.ndarray,
    classifier,
    *,
    prefix: str = "",
    spatial: bool = True,
    n_neighbors: int | None = None,
    n_classes: int | None = None,
) -> dict[str, float]:
    """Label generated and paired-real niches with a neutral classifier and
    compare.

    Args:
        gen_x / gen_pos: generated microenvironments ``(B, N, n_pcs)`` / ``(B, N, coord)``,
            centroid at point 0.
        gt_x / gt_pos: the **paired real target** microenvironments (same centroids), same shapes.
        classifier: a frozen, neutral ``torch.nn.Module`` (trained on the held-out slide). Spatial
            classifiers take the **expression-only** KNN point set ``(B, k+1, n_pcs)`` (centroid at
            point 0); the gene-only classifier takes the centroid expression ``(B, n_pcs)``.
        spatial: whether ``classifier`` is a spatial (microenvironment) net or gene-only.
        n_neighbors: KNN ``k`` for the spatial classifier. ``None`` -> use the value recorded on the
            classifier at training time (set by ``load_spatial_classifier``), so eval matches
            training; falls back to 10 with a warning if neither is available.
        n_classes: number of cell types; inferred from ``classifier.output_dim`` if not given.

    """
    import torch

    p = f"{prefix}/" if prefix else ""

    if spatial:
        k = _resolve_n_neighbors(n_neighbors, classifier)
        feats_gen = build_microenv_points(gen_x, gen_pos, k)
        feats_real = build_microenv_points(gt_x, gt_pos, k)
    else:
        feats_gen, feats_real = gen_x[:, 0, :], gt_x[:, 0, :]

    classifier.eval()
    device = next(classifier.parameters()).device
    with torch.no_grad():
        y_gen = classifier(torch.as_tensor(feats_gen, dtype=torch.float32, device=device))
        y_real = classifier(torch.as_tensor(feats_real, dtype=torch.float32, device=device))
        y_gen = y_gen.argmax(dim=-1).cpu().numpy()
        y_real = y_real.argmax(dim=-1).cpu().numpy()

    if n_classes is None:
        n_classes = int(getattr(classifier, "output_dim", int(max(y_gen.max(), y_real.max())) + 1))

    p_gen = proportions(y_gen, n_classes)
    p_real = proportions(y_real, n_classes)

    return {
        # Agreement: real labels are the target, so weighted-F1 weights by the real composition.
        f"{p}ct/f1": float(f1_score(y_real, y_gen, average="weighted", zero_division=0)),
        f"{p}ct/acc": float(accuracy_score(y_real, y_gen)),
        # Composition: divergence between the two label-proportion histograms.
        f"{p}ct/prop_kl": kl_divergence(p_real, p_gen),
        f"{p}ct/prop_tv": total_variation(p_real, p_gen),
        f"{p}ct/prop_jsd": jensen_shannon(p_real, p_gen),
    }
