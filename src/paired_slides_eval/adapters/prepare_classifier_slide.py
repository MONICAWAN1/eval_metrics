"""Prepare a neutral classifier-training slide for the standalone ``evaluate`` / ``otcfm`` path.

Data-prep only (no training) — the standalone counterpart of NicheFlow's ``prepare_abca.py
--classifier-slide``. It writes the preprocessed-slide ``.pkl`` that the Hydra classifier trainer
(``python -m paired_slides_eval.classifier.train data.datamodule.data_fp=<this.pkl>``) consumes.

A held-out slide (neither source nor target) is projected into the **same** feature space + label
vocabulary that ``evaluate`` uses on the otcfm path: a PCA fit on the
target's raw genes (``--n_pcs``, un-whitened) + the target's cell-type order, with **raw** spatial
coordinates. Training on exactly that representation makes the classifier neutral and directly
applicable to the target & generated niches at eval time. All the niche preprocessing is the ported
adapter code (``adapters.nicheflow.preprocess.preprocess_classifier_slide_into_pca``).

Usage::

    python -m paired_slides_eval.adapters.prepare_classifier_slide \
        --target data/adata_..-1.001.h5ad \
        --classifier_slide data/adata_..-1.002.h5ad \
        --ct_key class --n_pcs 50 --out data/abca_1.002_clf_otcfm.pkl
"""

from __future__ import annotations

import pickle
from pathlib import Path


def prepare_classifier_slide(
    target_h5ad,
    classifier_slide_h5ad,
    *,
    ct_key: str = "class",
    n_pcs: int = 50,
    spatial_key: str = "spatial",
    radius: float = 0.15,
    dx: float = 0.15,
    dy: float = 0.2,
    device: str = "cpu",
):
    """Build the classifier-slide dataclass projected into the target's raw-gene PCA + label space.

    Returns the ``H5ADDatasetDataclass`` (pickle it to get the trainer's ``data_fp``).
    """
    from paired_slides_eval.adapters.nicheflow.preprocess import (
        preprocess_classifier_slide_into_pca,
    )
    from paired_slides_eval.contract import TargetSlide
    from paired_slides_eval.data.anndata import cell_type_labels, read_anndata

    # Fit the target exactly as evaluate.py does: PCA on the target's raw genes + its label order.
    target = TargetSlide.from_anndata(
        target_h5ad, ct_key=ct_key, spatial_key=spatial_key, n_pcs=n_pcs
    )
    if target.pca is None:
        raise ValueError("Pass n_pcs so the target PCA exists; the classifier trains in it.")
    target_adata = read_anndata(target_h5ad)
    _, ct_to_int = cell_type_labels(target_adata, ct_key)
    if ct_to_int is None:
        raise ValueError(f"ct_key {ct_key!r} not found on the target; needed for the classifier.")

    return preprocess_classifier_slide_into_pca(
        classifier_slide_h5ad,
        target.pca,
        target_adata.var_names,
        list(ct_to_int),  # category order == eval's label mapping
        cell_type_column=ct_key,
        radius=radius,
        dx=dx,
        dy=dy,
        device=device,
    )


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Prepare a classifier-training slide .pkl for the Hydra classifier trainer."
    )
    ap.add_argument("--target", required=True, help="target slide .h5ad (defines PCA + labels)")
    ap.add_argument(
        "--classifier_slide",
        required=True,
        help="held-out slide .h5ad to train on (close but different from the target)",
    )
    ap.add_argument("--ct_key", default="class", help="obs column with cell types")
    ap.add_argument("--n_pcs", type=int, default=50, help="PCA components (must match evaluate)")
    ap.add_argument("--spatial_key", default="spatial")
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--dx", type=float, default=0.15)
    ap.add_argument("--dy", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True, help="path to write the .pkl (the trainer's data_fp)")
    args = ap.parse_args()

    ds = prepare_classifier_slide(
        args.target,
        args.classifier_slide,
        ct_key=args.ct_key,
        n_pcs=args.n_pcs,
        spatial_key=args.spatial_key,
        radius=args.radius,
        dx=args.dx,
        dy=args.dy,
        device=args.device,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
    n_cells = len(ds.ct)
    print(
        f"saved classifier-slide dataclass -> {out} "
        f"({n_cells} cells, {len(ds.ct_ordered)} classes, n_pcs={ds.X_pca.shape[1]})"
    )


if __name__ == "__main__":
    _main()
