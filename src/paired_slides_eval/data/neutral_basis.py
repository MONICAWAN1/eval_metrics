"""The model-neutral evaluation basis ``P*`` — an unwhitened PCA on the target's log-gene.

The fair space every model is scored in (see ``docs/neutral_basis_eval_plan.md``). Deliberately
adapter-free: it only touches log-gene arrays + a dataclass, so the basis (and its scree-knee
dimension ``k``) is a property of the **target slide**, not of any generator. ``prepare_shared_slides``
fits it and attaches it to the shared pickle; NicheFlow's ``preprocess_pair`` no longer knows about it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paired_slides_eval.data.shared_pca import SharedGenePCA


def pca_knee(explained_variance, k_min: int = 5, k_max: int = 50) -> int:
    """Scree elbow: the k at max distance from the first→last chord, clamped to ``[k_min, k_max]``.

    Parameter-free knee for the headline neutral-basis dimension (~past the noise floor).
    """
    ev = np.asarray(explained_variance, dtype=np.float64).ravel()
    n = len(ev)
    if n <= k_min:
        return int(n)
    x = np.arange(n, dtype=np.float64)
    x1, y1, x2, y2 = x[0], ev[0], x[-1], ev[-1]
    dist = np.abs((y2 - y1) * x - (x2 - x1) * ev + x2 * y1 - y2 * x1) / (np.hypot(y2 - y1, x2 - x1) + 1e-12)
    return int(np.clip(int(np.argmax(dist)) + 1, k_min, min(k_max, n)))


@dataclass
class NeutralBasis:
    """Frozen ``P*``: an unwhitened PCA (``pcs``, ``mean``) fit on the target log-gene + headline ``k``."""

    pcs: np.ndarray  # (n_genes, n_pcs) loadings
    mean: np.ndarray  # (n_genes,) log-gene centering mean
    target_sum: float  # τ the log-gene was normalised to (shared with the pair, for the guard)
    k: int  # scree-knee headline dimension
    var_names: list | None = None

    def project(self, log_gene: np.ndarray) -> np.ndarray:
        """Log-gene ``(cells, n_genes)`` -> unwhitened ``P*`` scores ``(cells, n_pcs)``."""
        return ((np.asarray(log_gene, dtype=np.float64) - self.mean) @ self.pcs).astype(np.float32)


def fit_neutral_basis(log_gene, target_idx, *, n_pcs, target_sum, var_names=None):
    """Fit ``P*`` on the target rows of a pair's log-gene; return ``(NeutralBasis, all-cell scores)``.

    ``log_gene`` is the whole pair (row order matching ``X_pca``); the PCA is fit on ``target_idx``
    only, then every cell is projected so the scores index like the rest of the dataclass.
    """
    from sklearn.decomposition import PCA

    log_gene = np.asarray(log_gene, dtype=np.float64)
    target_idx = np.asarray(target_idx)
    k_fit = min(int(n_pcs), len(target_idx) - 1, log_gene.shape[1])
    p = PCA(n_components=k_fit).fit(log_gene[target_idx])
    basis = NeutralBasis(
        pcs=p.components_.T.astype(np.float64),
        mean=p.mean_.astype(np.float64),
        target_sum=float(target_sum),
        k=pca_knee(p.explained_variance_),
        var_names=list(var_names) if var_names is not None else None,
    )
    return basis, basis.project(log_gene)


def attach_neutral_basis(ds, basis: NeutralBasis, scores) -> None:
    """Persist a fitted ``P*`` + its per-cell scores onto a dataclass (pair or classifier slide)."""
    ds.neutral_pcs = basis.pcs
    ds.neutral_mean = basis.mean
    ds.neutral_target_sum = basis.target_sum
    ds.neutral_k = basis.k
    ds.neutral_x = np.asarray(scores, dtype=np.float32)


def neutral_basis_from_dataclass(ds, *, k: int | None = None, apply_lognorm: bool = True) -> SharedGenePCA:
    """Rebuild ``P*`` from a pickle as a whitening-off :class:`SharedGenePCA` (centre+project only).

    Sliced to ``k`` (default ``ds.neutral_k`` headline). Raises if the pickle predates the fields.
    """
    if getattr(ds, "neutral_pcs", None) is None:
        raise ValueError("Pickle has no neutral basis (P*); re-run prepare_shared_slides to add it.")
    full = np.asarray(ds.neutral_pcs, dtype=np.float64)  # (n_genes, n_pcs)
    k = int(k or getattr(ds, "neutral_k", None) or full.shape[1])
    k = min(k, full.shape[1])
    return SharedGenePCA(
        pcs=full[:, :k],
        lognorm_mean=np.asarray(ds.neutral_mean, dtype=np.float64).ravel(),
        xpca_mean=np.zeros(k, dtype=np.float64),
        xpca_std=np.ones(k, dtype=np.float64),
        target_sum=float(ds.neutral_target_sum),
        var_names=list(ds.var_names) if getattr(ds, "var_names", None) is not None else None,
        apply_lognorm=apply_lognorm,
    )
