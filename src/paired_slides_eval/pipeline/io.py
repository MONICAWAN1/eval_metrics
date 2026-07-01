"""Serialization for generated cells.

``write_generated`` writes a :class:`~paired_slides_eval.contract.GeneratedNiches` or
:class:`~paired_slides_eval.contract.GeneratedSlide` to ``.h5ad`` or ``.npz`` in the same layout the
evaluator's loader reads (:func:`paired_slides_eval.evaluate._load_generated`), so a generate step
and a later evaluate step compose.
"""

from __future__ import annotations

import os

import numpy as np

from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide


def _source_pca_arrays(generated: GeneratedNiches | GeneratedSlide) -> dict | None:
    """Serialise ``generated.source_pca`` (a ``GenPCAInversion``) to a dict of arrays, or ``None``.

    Persisting these lets a standalone evaluate step reproject model-native reduced cells (e.g.
    OT-CFM's whitened PCA) into the shared basis without the original checkpoint.
    """
    sp = getattr(generated, "source_pca", None)
    if sp is None:
        return None
    out = {
        "components": np.asarray(sp.components, dtype=np.float32),
        "mean": np.asarray(sp.mean, dtype=np.float32),
        "sc_mean": np.asarray(sp.sc_mean, dtype=np.float32),
        "sc_scale": np.asarray(sp.sc_scale, dtype=np.float32),
    }
    if getattr(sp, "var_names", None) is not None:
        out["var_names"] = np.asarray([str(v) for v in sp.var_names])
    return out


def read_source_pca(mapping) -> object | None:
    """Rebuild a ``GenPCAInversion`` from a persisted mapping, or ``None`` if absent.

    The inverse of :func:`_source_pca_arrays`. Accepts either an ``adata.uns['source_pca']``
    sub-dict (keys ``components``/``mean``/...) or a flat mapping using the ``source_pca__<key>``
    prefix (the ``.npz`` layout / an unpickled dict).
    """
    keys = list(getattr(mapping, "files", None) or mapping.keys())
    if "source_pca__components" in keys:
        d = {k[len("source_pca__"):]: mapping[k] for k in keys if k.startswith("source_pca__")}
    elif "components" in keys:
        d = {k: mapping[k] for k in keys}
    else:
        return None

    from paired_slides_eval.data.shared_pca import GenPCAInversion

    var_names = d.get("var_names")
    if var_names is not None:
        var_names = [str(v) for v in np.asarray(var_names).ravel().tolist()]
    return GenPCAInversion(
        components=np.asarray(d["components"], dtype=np.float64),
        mean=np.asarray(d["mean"], dtype=np.float64).ravel(),
        sc_mean=np.asarray(d["sc_mean"], dtype=np.float64).ravel(),
        sc_scale=np.asarray(d["sc_scale"], dtype=np.float64).ravel(),
        var_names=var_names,
    )


def write_generated(generated: GeneratedNiches | GeneratedSlide, path: str) -> str:
    """Write generated cells to ``path`` (``.h5ad`` or ``.npz``); returns ``path``.

    ``.h5ad`` — niche-shaped cells become flat rows with ``obs['niche_id']`` grouping each niche
    (centroid first), coordinates in ``obsm['spatial']``, and any paired ground truth in
    ``obsm['gt_x']`` / ``obsm['gt_pos']`` / ``obs['gt_ct']``; a flat slide is ``X`` +
    ``obsm['spatial']``. ``.npz`` — niche-shaped: 3-D ``x`` / ``pos`` (+ optional ``gt_x`` /
    ``gt_pos`` / ``gt_ct``); flat: 2-D ``x`` / ``pos``.
    """
    path = str(path)
    # Create parent dirs so nested default output paths (e.g. artifacts/<model>/...) work.
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if path.endswith(".npz"):
        _write_npz(generated, path)
    elif path.endswith(".h5ad"):
        _write_h5ad(generated, path)
    else:
        raise ValueError(f"Unsupported output extension for {path!r}; use '.h5ad' or '.npz'.")
    return path


def _write_npz(generated: GeneratedNiches | GeneratedSlide, path: str) -> None:
    arrays = {"x": np.asarray(generated.x), "pos": np.asarray(generated.pos)}
    if isinstance(generated, GeneratedNiches):
        for key in ("gt_x", "gt_pos", "gt_ct"):
            value = getattr(generated, key, None)
            if value is not None:
                arrays[key] = np.asarray(value)
    source_pca = _source_pca_arrays(generated)
    if source_pca is not None:
        for key, value in source_pca.items():
            arrays[f"source_pca__{key}"] = value
    np.savez(path, **arrays)


def _write_h5ad(generated: GeneratedNiches | GeneratedSlide, path: str) -> None:
    import anndata as ad
    import pandas as pd

    if isinstance(generated, GeneratedNiches):
        b, n, _ = generated.x.shape
        adata = ad.AnnData(X=generated.x.reshape(b * n, -1).astype(np.float32))
        adata.obs["niche_id"] = np.repeat(np.arange(b), n)
        adata.obsm["spatial"] = generated.pos.reshape(b * n, -1).astype(np.float32)
        if generated.gt_x is not None and generated.gt_pos is not None:
            adata.obsm["gt_x"] = generated.gt_x.reshape(b * n, -1).astype(np.float32)
            adata.obsm["gt_pos"] = generated.gt_pos.reshape(b * n, -1).astype(np.float32)
        if generated.gt_ct is not None:
            adata.obs["gt_ct"] = np.repeat(np.asarray(generated.gt_ct), n)
            adata.obs = adata.obs.astype({"gt_ct": "int64"})
        adata.obs = adata.obs.astype({"niche_id": "int64"})
        adata.obs.index = pd.RangeIndex(b * n).astype(str)
    else:  # GeneratedSlide
        adata = ad.AnnData(X=np.asarray(generated.x, dtype=np.float32))
        adata.obsm["spatial"] = np.asarray(generated.pos, dtype=np.float32)
        adata.obs.index = pd.RangeIndex(generated.x.shape[0]).astype(str)
    source_pca = _source_pca_arrays(generated)
    if source_pca is not None:
        adata.uns["source_pca"] = source_pca
    adata.write_h5ad(path)
