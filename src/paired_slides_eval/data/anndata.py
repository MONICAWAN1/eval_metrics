"""Read evaluation inputs straight from original AnnData (``.h5ad``) files.

The user-facing inputs are plain AnnData slides containing raw gene expression + spatial coordinates. 
This module pulls the arrays the metrics need out of an AnnData file:

* expression — ``adata.X`` (the genes) by default, or an ``obsm``/``layers`` key if you already
  have a reduced space (e.g. ``X_pca``);
* coordinates — ``adata.obsm[spatial_key]`` (squidpy/scanpy convention, default ``"spatial"``);
* optional cell-type labels — ``adata.obs[ct_key]`` (mapped to ints).

PCA is not assumed to exist; pass ``n_pcs`` to fit one on the target (see :func:`fit_pca`) and project 
both sides into it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def read_anndata(adata_or_path):
    """Return an ``AnnData`` from either an in-memory object or a path to a ``.h5ad`` file."""
    if isinstance(adata_or_path, (str, bytes)) or hasattr(adata_or_path, "__fspath__"):
        import anndata as ad

        return ad.read_h5ad(str(adata_or_path))
    return adata_or_path


def _densify(x) -> np.ndarray:
    """Materialise a (possibly sparse) AnnData matrix as a dense ``float32`` array."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def slide_expression(adata, expr_key: str | None = None) -> np.ndarray:
    """Pull the expression matrix: ``adata.X`` (default) or an ``obsm``/``layers`` key."""
    if expr_key is None:
        return _densify(adata.X)
    if expr_key in adata.obsm:
        return _densify(adata.obsm[expr_key])
    if expr_key in adata.layers:
        return _densify(adata.layers[expr_key])
    raise KeyError(
        f"expr_key {expr_key!r} not found in adata.obsm or adata.layers "
        f"(obsm: {list(adata.obsm)}, layers: {list(adata.layers)})"
    )


def slide_coords(adata, spatial_key: str = "spatial") -> np.ndarray:
    """Pull the spatial coordinates from ``adata.obsm[spatial_key]``."""
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"spatial_key {spatial_key!r} not found in adata.obsm (have: {list(adata.obsm)}). "
            f"Set spatial_key to the obsm entry holding the coordinates."
        )
    return np.asarray(adata.obsm[spatial_key], dtype=np.float32)


def cell_type_labels(adata, ct_key: str | None) -> tuple[np.ndarray | None, dict | None]:
    """Map ``adata.obs[ct_key]`` to integer labels, returning ``(labels, ct_to_int)``.

    Uses the categorical's full ``cat.categories`` order when available (so the mapping is stable
    even if some types are absent from a given slide), else the sorted unique values. Returns
    ``(None, None)`` when ``ct_key`` is ``None`` or missing.
    """
    if ct_key is None or ct_key not in adata.obs:
        return None, None
    col = adata.obs[ct_key]
    if hasattr(col, "cat"):
        categories = list(col.cat.categories)
    else:
        categories = sorted(map(str, np.unique(np.asarray(col))))
    ct_to_int = {c: i for i, c in enumerate(categories)}
    labels = np.array([ct_to_int[str(v)] for v in np.asarray(col).astype(str)], dtype=np.int64)
    return labels, ct_to_int


@dataclass
class _PCA:
    """A minimal frozen PCA: centre on the fit mean, project onto stored components.

    Mirrors ``sklearn``'s ``transform`` (centre then ``@ components_.T``) so a PCA fit on the
    target slide can be applied identically to the generated cells, guaranteeing both live in
    one basis. Kept tiny and dependency-light so it pickles into a results bundle cleanly.
    """

    mean: np.ndarray
    components: np.ndarray  # (n_pcs, n_features)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return ((x - self.mean) @ self.components.T).astype(np.float32)


def fit_pca(x: np.ndarray, n_pcs: int) -> _PCA:
    """Fit a PCA on ``x`` (rows = cells) and return a frozen, re-applicable transform.

    Uses ``sklearn.decomposition.PCA`` for the fit, then keeps only the mean + components so the
    same projection can be applied to generated cells (``pca.transform``).
    """
    from sklearn.decomposition import PCA

    x = np.asarray(x, dtype=np.float64)
    n_pcs = min(n_pcs, x.shape[1], x.shape[0])
    model = PCA(n_components=n_pcs).fit(x)
    return _PCA(
        mean=model.mean_.astype(np.float64), components=model.components_.astype(np.float64)
    )
