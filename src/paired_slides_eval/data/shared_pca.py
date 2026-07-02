"""Reconstruct NicheFlow's gene -> X_pca recipe as a re-applicable transform.

NicheFlow fits its shared PCA as ``normalize_total -> log1p -> PCA(source+target) -> whiten`` and
standardises coordinates per slide (see
:func:`~paired_slides_eval.adapters.nicheflow.preprocess.preprocess_pair`). Its generated cells are
emitted *already* in that whitened ``X_pca`` + standardised-coord space. A model that instead emits
**gene-space** cells (the OT-CFM baseline) must be projected into the *exact same* basis to be
comparable — that is what :class:`SharedGenePCA` does, replayed from the stats persisted on the
preprocessed-slide pickle.

:class:`SharedGenePCA` is duck-typed to the minimal ``_PCA`` interface
(:mod:`paired_slides_eval.data.anndata`): it exposes ``components`` of shape ``(n_pcs, n_genes)``
and a ``transform`` taking genes -> ``(cells, n_pcs)``. So
:func:`~paired_slides_eval.contract._pca_aware_transform` detects gene-space cells (width == n_genes)
and projects them, while already-reduced cells (width == n_pcs) pass through — both
:meth:`~paired_slides_eval.contract.GeneratedNiches.project` paths work unchanged.

Pure numpy (no scanpy/torch): ``normalize_total`` + ``log1p`` are reimplemented so projecting new
cells needs none of the heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SharedGenePCA:
    """The frozen gene -> whitened-X_pca transform, re-applicable to new gene-space cells.

    Attributes:
        pcs: the PCA loadings ``(n_genes, n_pcs)`` (``varm['PCs']``), so ``X_pca = X_centered @ pcs``.
        lognorm_mean: ``(n_genes,)`` per-gene mean of the log-normalised fit data (PCA centering).
        xpca_mean / xpca_std: ``(n_pcs,)`` whitening stats applied after projection.
        target_sum: the ``normalize_total`` scale (median of fit per-cell totals).
        var_names: gene-panel order the PCA was fit on (for aligning input columns); may be ``None``.
        apply_lognorm: run ``normalize_total + log1p`` before centering. Set ``False`` only when the
            input cells are *already* log-normalised (e.g. a model whose decoder emits lognorm
            expression) — re-applying it would double-transform..
    """

    pcs: np.ndarray  # (n_genes, n_pcs)
    lognorm_mean: np.ndarray  # (n_genes,)
    xpca_mean: np.ndarray  # (n_pcs,)
    xpca_std: np.ndarray  # (n_pcs,)
    target_sum: float
    var_names: list | None = None
    apply_lognorm: bool = True

    @property
    def components(self) -> np.ndarray:
        """``(n_pcs, n_genes)`` — the ``_PCA``-style components used for space auto-detection."""
        return np.asarray(self.pcs, dtype=np.float64).T

    def align_genes(self, genes: np.ndarray, input_var_names) -> np.ndarray:
        """Reorder ``genes`` columns to the fit ``var_names`` (no-op when already aligned / unknown)."""
        if self.var_names is None or input_var_names is None:
            return genes
        import pandas as pd

        idx = pd.Index([str(v) for v in input_var_names]).get_indexer(
            [str(v) for v in self.var_names]
        )
        if (idx < 0).any():
            missing = [str(v) for v, j in zip(self.var_names, idx) if j < 0][:5]
            raise ValueError(f"Input is missing fit genes (e.g. {missing}); align the panel first.")
        return np.asarray(genes)[:, idx]

    def transform(self, genes: np.ndarray) -> np.ndarray:
        """Project ``genes`` ``(cells, n_genes)`` (in fit ``var_names`` order) into whitened X_pca."""
        x = np.asarray(genes, dtype=np.float64)
        if self.apply_lognorm:
            totals = x.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1.0
            x = np.log1p(x / totals * self.target_sum)
        return self.transform_lognorm(x)

    def transform_lognorm(self, genes: np.ndarray) -> np.ndarray:
        """Project **already log-normalised** ``genes`` ``(cells, n_genes)`` into whitened X_pca.

        The ``normalize_total + log1p`` head of :meth:`transform` is skipped — use this when the
        input is already in the log-normalised gene space (e.g. a model-native PCA inverted back to
        log-gene by :class:`GenPCAInversion`), so it is not double-log-normalised.
        """
        x = np.asarray(genes, dtype=np.float64) - self.lognorm_mean
        x = x @ np.asarray(self.pcs, dtype=np.float64)  # (cells, n_pcs)
        x = (x - self.xpca_mean) / self.xpca_std
        return x.astype(np.float32)


def shared_pca_from_dataclass(ds, *, apply_lognorm: bool = True) -> SharedGenePCA:
    """Build a :class:`SharedGenePCA` from a preprocessed-slide dataclass (or raise if it can't).

    Needs the recipe fields ``preprocess_pair`` now persists (``lognorm_mean``,
    ``lognorm_target_sum``, ``PCs``, ``stats['X_pca']``). Pickles predating those carry ``None`` and
    must be re-preprocessed.
    """
    missing = [
        name
        for name in ("lognorm_mean", "lognorm_target_sum", "PCs")
        if getattr(ds, name, None) is None
    ]
    xpca = ds.stats.get("X_pca") if getattr(ds, "stats", None) else None
    if not xpca or "mean" not in xpca or "std" not in xpca:
        missing.append("stats['X_pca']")
    if missing:
        raise ValueError(
            "This preprocessed-slide pickle cannot back a SharedGenePCA (missing "
            f"{', '.join(missing)}). Re-run preprocess_pair so the recipe stats are persisted."
        )
    return SharedGenePCA(
        pcs=np.asarray(ds.PCs, dtype=np.float64),
        lognorm_mean=np.asarray(ds.lognorm_mean, dtype=np.float64).ravel(),
        xpca_mean=np.asarray(xpca["mean"], dtype=np.float64).ravel(),
        xpca_std=np.asarray(xpca["std"], dtype=np.float64).ravel(),
        target_sum=float(ds.lognorm_target_sum),
        var_names=list(ds.var_names) if getattr(ds, "var_names", None) is not None else None,
        apply_lognorm=apply_lognorm,
    )


@dataclass
class GenPCAInversion:
    """A generative model's own PCA -> log-normalised-gene inverse, carried with its cells.

    Two PCA fits share no direct linear map, only the log-gene space they were both fit on. So a
    model emitting its own reduced PCA (OT-CFM, NicheFlow) carries this inverse; eval un-projects to
    log-gene, then forwards through the neutral basis. ``k`` is read off ``components`` — never
    hard-coded.

    Attributes:
        components: ``(k, n_genes)`` loadings (sklearn ``components_``): ``log_gene = scores @ components + mean``.
        mean: ``(n_genes,)`` PCA centering mean (per-gene mean of the model's log-gene fit data).
        sc_mean / sc_scale: ``(k,)`` un-whitening stats (``scores = z * sc_scale + sc_mean``); 0/1 if unwhitened.
        var_names: the model's gene-panel order, for aligning to the neutral basis' panel; ``None`` if unknown.
        target_sum: the model's ``normalize_total`` scale (τ), for the cross-model log-norm consistency guard.
    """

    components: np.ndarray  # (k, n_genes)
    mean: np.ndarray  # (n_genes,)
    sc_mean: np.ndarray  # (k,)
    sc_scale: np.ndarray  # (k,)
    var_names: list | None = None
    target_sum: float | None = None

    @property
    def n_pcs(self) -> int:
        """``k`` — the model-native reduced dimension, used to auto-detect this space by width."""
        return int(np.asarray(self.components).shape[0])

    @property
    def n_genes(self) -> int:
        """``n_genes`` — the model's gene-panel size (width of the reconstructed log-gene cells)."""
        return int(np.asarray(self.components).shape[1])

    def to_log_gene(self, z: np.ndarray) -> np.ndarray:
        """Invert model-native PCA scores ``(cells, k)`` back to log-normalised gene expression.

        ``scores = z * sc_scale + sc_mean`` (un-whiten) then ``scores @ components + mean``
        (un-PCA). Linear and lossless up to the ``k``-component truncation; it stops at the
        log-normalised space (no ``expm1``), so no raw counts are ever fabricated.
        """
        z = np.asarray(z, dtype=np.float64)
        scores = z * np.asarray(self.sc_scale, dtype=np.float64) + np.asarray(
            self.sc_mean, dtype=np.float64
        )
        gene = scores @ np.asarray(self.components, dtype=np.float64) + np.asarray(
            self.mean, dtype=np.float64
        )
        return gene.astype(np.float32)


@dataclass
class CoordStandardizer:
    """Per-slide coordinate standardisation ``(pos - mean) / std``, re-applicable to new coords.

    Mirrors :meth:`H5ADPreprocessor._normalize_coordinates_and_features`, so a model emitting **raw**
    coordinates (the OT-CFM baseline) can be mapped into the target's standardised frame that the
    niche model and the classifier were trained in.
    """

    mean: np.ndarray
    std: np.ndarray

    def transform(self, pos: np.ndarray) -> np.ndarray:
        return ((np.asarray(pos, dtype=np.float64) - self.mean) / self.std).astype(np.float32)


def coord_standardizer_from_dataclass(ds, timepoint: str | None = None) -> CoordStandardizer:
    """Build a :class:`CoordStandardizer` from the dataclass's ``stats['coords'][timepoint]``.

    ``timepoint`` defaults to the target slide (last in ``timepoints_ordered``). Raises if the
    pickle was built with min-max scaling (``standardize_coordinates=False``) instead.
    """
    tp = timepoint or ds.timepoints_ordered[-1]
    cstats = ds.stats.get("coords", {}).get(tp) if getattr(ds, "stats", None) else None
    if not cstats or "mean" not in cstats or "std" not in cstats:
        raise ValueError(
            f"No standardised coord stats for timepoint {tp!r}; the pickle must be built with "
            "standardize_coordinates=True to reconcile new coordinates into its frame."
        )
    return CoordStandardizer(
        mean=np.asarray(cstats["mean"], dtype=np.float64).ravel(),
        std=np.asarray(cstats["std"], dtype=np.float64).ravel(),
    )
