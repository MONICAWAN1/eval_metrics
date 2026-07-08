"""Classifier two-sample test (C2ST) — a label-free distribution test.

Train a binary classifier to tell sample ``X`` (label 0) from ``Y`` (label 1); the held-out
accuracy/AUC is the statistic. ~0.5 == indistinguishable (good); ~1.0 == trivially separable
(bad). ``c2st`` / ``c2st_significance`` are the framework-free kernels (follow Lopez-Paz &
Oquab 2017); ``c2st_metrics`` is the wrapper computing the per-cell joint and expression-only views.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.neural_network import MLPClassifier

from paired_slides_eval.metrics._common import subsample


def c2st(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 0,
    n_folds: int = 5,
    z_score: bool = True,
    max_iter: int = 1000,
) -> tuple[float, float]:
    """Return ``(accuracy, roc_auc)`` of an MLP trained to separate ``X`` from ``Y``.

    Both metrics are the mean over ``n_folds`` cross-validation folds; ~0.5 indistinguishable,
    ~1.0 trivially separable. ``X`` is the reference for z-scoring (pass the *real* sample).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if z_score:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X = (X - mean) / std
        Y = (Y - mean) / std

    ndim = X.shape[1]
    data = np.concatenate([X, Y], axis=0)
    target = np.concatenate([np.zeros(len(X)), np.ones(len(Y))])

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=max_iter,
        solver="adam",
        random_state=seed,
    )
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    res = cross_validate(clf, data, target, cv=cv, scoring=["accuracy", "roc_auc"])
    return float(res["test_accuracy"].mean()), float(res["test_roc_auc"].mean())


def c2st_significance(
    X: np.ndarray,
    Y: np.ndarray,
    n_perm: int = 100,
    seed: int = 0,
    n_folds: int = 3,
    z_score: bool = True,
    max_n: int = 1000,
    max_iter: int = 300,
) -> dict[str, float]:
    """Permutation significance test for a C2ST AUC.

    Computes the observed AUC for ``X`` (real) vs ``Y`` (generated), then builds a null by
    pooling the two samples and randomly re-assigning labels ``n_perm`` times. Returns the
    observed ``auc``, the ``pval`` (fraction of shuffles with AUC >= observed), and ``null_p95``
    (95th percentile of the null; observed > null_p95 means p < 0.05).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    kw = dict(seed=seed, n_folds=n_folds, z_score=z_score, max_iter=max_iter)
    obs = c2st(X, Y, **kw)[1]

    pool = np.concatenate([X, Y], axis=0)
    n_x = len(X)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(pool))
        null[i] = c2st(pool[perm[:n_x]], pool[perm[n_x:]], **kw)[1]

    return {
        "auc": float(obs),
        "pval": float((1 + np.sum(null >= obs)) / (n_perm + 1)),
        "null_p95": float(np.percentile(null, 95)),
    }


def c2st_metrics(
    real_x: np.ndarray,
    real_pos: np.ndarray,
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    *,
    prefix: str = "",
    max_n: int = 2000,
    n_folds: int = 5,
    n_perm: int = 0,
    seed: int = 0,
) -> dict[str, float]:
    """Label-free C2ST across two views: per-cell joint ``[x|pos]`` and expr-only.

    The per-cell joint test detects a wrong expression<->position coupling the separate
    MMD/EMD marginals cannot; the expression-only (``gene_*``) test is a diagnostic for whether
    expression alone drives a separable joint.
    """
    p = f"{prefix}/" if prefix else ""
    rng = np.random.default_rng(seed)

    # Primary: per-cell joint [x | pos] (co-generation).
    real_joint = subsample(np.concatenate([real_x, real_pos], axis=1), max_n, rng)
    gen_joint = subsample(np.concatenate([gen_x, gen_pos], axis=1), max_n, rng)
    acc, auc = c2st(real_joint, gen_joint, seed=seed, n_folds=n_folds)

    # Diagnostic: per-cell expression-only (the gene marginal).
    gene_acc, gene_auc = c2st(
        subsample(real_x, max_n, rng), subsample(gen_x, max_n, rng), seed=seed, n_folds=n_folds
    )

    out = {
        f"{p}c2st/acc": acc,
        f"{p}c2st/auc": auc,
        f"{p}c2st/gene_acc": gene_acc,
        f"{p}c2st/gene_auc": gene_auc,
    }
    if n_perm > 0:
        sig = c2st_significance(real_joint, gen_joint, n_perm=n_perm, seed=seed)
        out[f"{p}c2st/sig_auc"] = sig["auc"]
        out[f"{p}c2st/sig_pval"] = sig["pval"]
        out[f"{p}c2st/sig_null_p95"] = sig["null_p95"]
    return out
