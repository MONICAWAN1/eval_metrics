"""Build the niche dataclass from raw AnnData slides — port of ``scripts/prepare_abca``.

Two raw ``.h5ad`` slides become the source (``A``) and target (``B``) of the flow. We compute one
**shared PCA** on the concatenated pair (``normalize_total -> log1p -> PCA``, exactly as
NicheFlow's ``prepare_abca._compute_pca``), then run :class:`H5ADPreprocessor` to standardize and
build the radius graph + grid subsample. A held-out classifier-training slide can be projected
into the same PCA basis + label space so a classifier trained on it applies to the target.

No global alignment / PASTE2: coordinates are standardized per slide, matching the original
unaligned NicheFlow path.
"""

from __future__ import annotations

import numpy as np

from paired_slides_eval.adapters.nicheflow.h5ad_preprocessor import H5ADPreprocessor
from paired_slides_eval.data.dataclass import H5ADDatasetDataclass

SLIDE_A = "A"  # source
SLIDE_B = "B"  # target


def _load_two_slides(slide_a, slide_b, slide_column: str):
    """Read two ``.h5ad`` slides, tag each with a ``slide`` id, and concatenate.

    Requires an identical gene panel across the two files (same ``var_names`` in the same order)
    so a single shared PCA basis is meaningful. (Port of ``prepare_abca._load_two_slides``.)
    """
    import anndata as ad

    a = ad.read_h5ad(slide_a)
    b = ad.read_h5ad(slide_b)

    if not a.var_names.equals(b.var_names):
        raise ValueError(
            "The two slides must share the exact same gene panel (var_names) in the same order. "
            f"Got {a.n_vars} vs {b.n_vars} genes; identical={a.var_names.equals(b.var_names)}."
        )

    a.obs[slide_column] = SLIDE_A
    b.obs[slide_column] = SLIDE_B
    adata = ad.concat([a, b], join="inner", merge="same", index_unique="-")
    adata.obs[slide_column] = adata.obs[slide_column].astype("category")
    return adata


def compute_pca(adata, n_pcs: int) -> None:
    """``normalize_total -> log1p -> PCA``, filling ``obsm['X_pca']`` and ``varm['PCs']``.

    Port of ``prepare_abca._compute_pca``. Raw counts are stashed in ``layers['counts']`` first;
    PCA is fit on the concatenated data so both slides share one PC basis.
    """
    import scanpy as sc

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=n_pcs)  # -> obsm['X_pca'], varm['PCs']
    assert "X_pca" in adata.obsm and "PCs" in adata.varm


def preprocess_pair(
    source_h5ad,
    target_h5ad,
    *,
    n_pcs: int = 50,
    cell_type_column: str = "class",
    slide_column: str = "slide",
    radius: float = 0.15,
    dx: float = 0.15,
    dy: float = 0.2,
    device: str = "cpu",
) -> tuple[H5ADDatasetDataclass, H5ADPreprocessor]:
    """Preprocess a (source, target) pair of raw slides into the niche dataclass.

    Returns ``(dataclass, preprocessor)`` — the preprocessor is returned so its ``ct_ordered`` and
    ``X_pca`` stats can be reused to project a classifier slide (see
    :func:`preprocess_classifier_slide`). ``timepoints_ordered`` is ``["A", "B"]`` (source, target).
    """
    adata = _load_two_slides(source_h5ad, target_h5ad, slide_column)
    compute_pca(adata, n_pcs)

    pre = H5ADPreprocessor(
        timepoint_column=slide_column,
        cell_type_column=cell_type_column,
        timepoints_ordered=[SLIDE_A, SLIDE_B],
        standardize_coordinates=True,
        radius=radius,
        dx=dx,
        dy=dy,
        device=device,
    )
    pre.preprocess_data(adata)
    # Stash the log-normalised mean so a classifier slide can be projected into this PCA basis.
    pre._lognorm_mean = np.asarray(adata.X.mean(axis=0)).ravel()
    pre._pcs = np.asarray(adata.varm["PCs"])
    pre._var_names = adata.var_names
    return pre.to_dataclass(), pre


def preprocess_classifier_slide_into_pca(
    classifier_h5ad,
    pca,
    var_names,
    ct_ordered,
    *,
    cell_type_column: str = "class",
    slide_column: str = "slide",
    radius: float = 0.15,
    dx: float = 0.15,
    dy: float = 0.2,
    device: str = "cpu",
) -> H5ADDatasetDataclass:
    """Project a held-out classifier slide into an **already-fit raw-gene PCA** + label space.

    Counterpart of :func:`preprocess_classifier_slide` for the standalone ``evaluate`` / ``otcfm``
    path (no NicheFlow shared preprocessing). There the target's feature space is a plain PCA fit on
    the target's raw genes (``TargetSlide.from_anndata`` with ``n_pcs``)
    and the generated/target cells are fed to the classifier with **un-standardised** ``X_pca`` and
    **raw** coordinates. So the classifier must train on exactly that: this projects the slide's raw
    genes through ``pca`` (no normalize/log, no per-component whitening) and keeps raw coordinates
    (the radius graph + niche neighbours are built on the raw frame), so the niche representation
    matches what :func:`paired_slides_eval.evaluate.evaluate` feeds at eval time.

    Args:
        classifier_h5ad: held-out slide (raw genes + ``obsm['spatial']``).
        pca: the target's frozen PCA (``TargetSlide.pca``); its ``transform`` projects raw genes.
        var_names: the target's gene panel order — the slide is reordered to it before projecting.
        ct_ordered: the target's cell-type vocabulary (ordered); cells outside it are dropped so the
            classifier's label space matches the target exactly.
    """
    import anndata as ad

    c = ad.read_h5ad(classifier_h5ad)
    if not c.var_names.equals(var_names):
        c = c[:, var_names].copy()  # match the target's gene panel/order

    # Keep only cells whose type is in the target vocabulary (so labels map 1:1 to eval).
    vocab = set(ct_ordered)
    keep = np.array([str(v) in vocab for v in c.obs[cell_type_column].astype(str)])
    if not keep.all():
        c = c[keep].copy()

    genes = c.X.toarray() if hasattr(c.X, "toarray") else np.asarray(c.X)
    n_pcs = pca.components.shape[0]
    c.obsm["X_pca"] = np.asarray(pca.transform(genes))
    c.varm["PCs"] = np.asarray(pca.components, dtype=np.float64).T  # (genes, n_pcs); unused later
    c.obs[slide_column] = "C"
    c.obs[slide_column] = c.obs[slide_column].astype("category")

    clf_pre = H5ADPreprocessor(
        timepoint_column=slide_column,
        cell_type_column=cell_type_column,
        timepoints_ordered=["C"],
        standardize_coordinates=True,
        radius=radius,
        dx=dx,
        dy=dy,
        device=device,
        external_ct_ordered=list(ct_ordered),
        # Identity X_pca stats: feed the raw projection (no whitening), exactly as eval does.
        external_x_pca_stats={"mean": np.zeros(n_pcs), "std": np.ones(n_pcs)},
    )
    clf_pre.preprocess_data(c)

    # Eval builds niches from RAW target/generated coordinates, so train on the raw frame too:
    # restore raw coords and rebuild the radius-graph neighbours on them.
    clf_pre.coords = np.asarray(c.obsm["spatial"], dtype=np.float64).copy()
    clf_pre.stats["coords"] = {}
    clf_pre._compute_radius_graphs()
    return clf_pre.to_dataclass()


def preprocess_classifier_slide(
    classifier_h5ad,
    base_preprocessor: H5ADPreprocessor,
    *,
    cell_type_column: str = "class",
    slide_column: str = "slide",
    radius: float = 0.15,
    dx: float = 0.15,
    dy: float = 0.2,
    device: str = "cpu",
) -> H5ADDatasetDataclass:
    """Project a held-out classifier slide into the source+target PCA basis + label space.

    Port of ``prepare_abca``'s ``--classifier-slide`` block (minus alignment). Reuses the
    log-normalised mean + ``PCs`` from ``base_preprocessor`` (the source+target fit), and its
    ``ct_ordered`` / ``X_pca`` stats, so a classifier trained on this slide applies to the
    target & generated niches it is evaluated on.
    """
    import anndata as ad
    import scanpy as sc

    mean_ab = base_preprocessor._lognorm_mean
    pcs = base_preprocessor._pcs

    c = ad.read_h5ad(classifier_h5ad)
    if not c.var_names.equals(base_preprocessor._var_names):
        c = c[:, base_preprocessor._var_names].copy()  # match the flow's gene panel/order

    # Keep only cells whose type is in the source+target vocabulary, so labels map 1:1 to the
    # classifier's classes (a held-out slide can carry a cell type absent from the pair).
    vocab = set(base_preprocessor.ct_ordered)
    keep = np.array([str(v) in vocab for v in c.obs[cell_type_column].astype(str)])
    if not keep.all():
        c = c[keep].copy()

    c.layers["counts"] = c.X.copy()
    sc.pp.normalize_total(c)
    sc.pp.log1p(c)
    x_c = c.X.toarray() if hasattr(c.X, "toarray") else np.asarray(c.X)
    c.obsm["X_pca"] = (x_c - mean_ab) @ pcs
    c.varm["PCs"] = pcs
    c.obs[slide_column] = "C"
    c.obs[slide_column] = c.obs[slide_column].astype("category")

    clf_pre = H5ADPreprocessor(
        timepoint_column=slide_column,
        cell_type_column=cell_type_column,
        timepoints_ordered=["C"],
        standardize_coordinates=True,
        radius=radius,
        dx=dx,
        dy=dy,
        device=device,
        external_ct_ordered=base_preprocessor.ct_ordered,
        external_x_pca_stats=base_preprocessor.stats["X_pca"],
    )
    clf_pre.preprocess_data(c)
    return clf_pre.to_dataclass()
