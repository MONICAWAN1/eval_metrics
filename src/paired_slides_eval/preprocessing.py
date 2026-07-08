"""Preprocessing namespace: the one shared basis every model is scored in.

Re-exports the shared-PCA + coordinate-standardisation recipe, so all models generate and are scored
in the **same PCA space** with **coordinates standardised the same way**. The gene→``X_pca`` replay and coord frame
are light (numpy); the raw-slide builders (``preprocess_pair`` / ``preprocess_classifier_slide``)
pull scanpy and load **lazily** on first access (the ``[pipeline]`` extra).
"""

from __future__ import annotations

from paired_slides_eval.data.shared_pca import (
    Basis,
    CoordStandardizer,
    SharedGenePCA,
    coord_standardizer_from_dataclass,
    shared_pca_from_dataclass,
)


def fit_basis(source, target, *, n_pcs: int = 50, cell_type_column: str = "class", **kwargs) -> Basis:
    """Fit the shared :class:`~paired_slides_eval.data.shared_pca.Basis` on a source+target pair.

    Runs the shared-PCA + coordinate fit and returns just the reusable basis. (The full pair
    dataclass — with the niche scaffolding evaluation also needs — comes from ``preprocess_pair``;
    this wraps it so ``fit`` and ``replay`` share one implementation.) Needs the ``[pipeline]`` extra.
    """
    from paired_slides_eval.adapters.nicheflow.preprocess import preprocess_pair

    ds, _ = preprocess_pair(source, target, n_pcs=n_pcs, cell_type_column=cell_type_column, **kwargs)
    return Basis.from_dataclass(ds)


_LAZY = {"preprocess_pair", "preprocess_classifier_slide"}


def __getattr__(name):
    # Heavy (scanpy) builders load only when actually used.
    if name in _LAZY:
        from paired_slides_eval.adapters.nicheflow import preprocess as _p

        return getattr(_p, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Basis",
    "fit_basis",
    "SharedGenePCA",
    "CoordStandardizer",
    "shared_pca_from_dataclass",
    "coord_standardizer_from_dataclass",
    "preprocess_pair",
    "preprocess_classifier_slide",
]
