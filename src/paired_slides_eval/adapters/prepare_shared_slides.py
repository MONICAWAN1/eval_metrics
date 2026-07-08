"""Build the shared artifacts for the unified cross-model evaluation (NicheFlow
recipe).

Replaces the retired raw-gene-PCA ``prepare_classifier_slide``. Produces two pickles from one
``preprocess_pair`` fit, so every model is measured in the same basis (see
``docs/comparability_plan.md``):

* ``--out_pair`` — the source+target pair pickle = the one shared PCA basis + standardised coord
  frame. Used as ``evaluate --target <pair.pkl> --shared_pca`` (it carries the gene->X_pca recipe, so
  gene-space OT-CFM cells project into the same basis the niche models trained in).
* ``--out_classifier`` — the held-out classifier-training slide projected into that same basis. Used
  as the classifier trainer's ``data.datamodule.data_fp``.

Needs the ``[pipeline]`` extra (scanpy/torch), like generation and classifier training.

Usage::

    python -m paired_slides_eval.adapters.prepare_shared_slides \
        --source DATA/adata_..-1.000.h5ad --target DATA/adata_..-1.001.h5ad \
        --classifier_slide DATA/adata_..-1.002.h5ad --ct_key class --n_pcs 50 \
        --out_pair data/abca_pair.pkl --out_classifier data/abca_clf.pkl

"""

from __future__ import annotations

import pickle
from pathlib import Path


def _dump(ds, out: str) -> None:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _main() -> None:
    import argparse

    from paired_slides_eval.adapters.nicheflow.preprocess import (
        preprocess_classifier_slide,
        preprocess_pair,
    )

    ap = argparse.ArgumentParser(
        description="Build the shared pair + classifier-slide pickles (NicheFlow recipe) for the "
        "unified cross-model evaluation.",
    )
    ap.add_argument("--source", required=True, help="source slide .h5ad (defines the shared PCA)")
    ap.add_argument("--target", required=True, help="target slide .h5ad (the evaluate --target)")
    ap.add_argument(
        "--classifier_slide",
        required=True,
        help="held-out slide .h5ad to train the classifier on (close but different from the target)",
    )
    ap.add_argument("--ct_key", default="class", help="obs column with cell types")
    ap.add_argument("--n_pcs", type=int, default=50, help="shared-PCA components")
    ap.add_argument("--radius", type=float, default=0.15)
    ap.add_argument("--dx", type=float, default=0.15)
    ap.add_argument("--dy", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_pair", required=True, help="path for the source+target pair .pkl")
    ap.add_argument(
        "--out_classifier",
        required=True,
        help="path for the classifier-slide .pkl (trainer data_fp)",
    )
    args = ap.parse_args()

    ds_pair, pre = preprocess_pair(
        args.source,
        args.target,
        n_pcs=args.n_pcs,
        cell_type_column=args.ct_key,
        radius=args.radius,
        dx=args.dx,
        dy=args.dy,
        device=args.device,
    )

    clf_ds = preprocess_classifier_slide(
        args.classifier_slide,
        pre,
        cell_type_column=args.ct_key,
        radius=args.radius,
        dx=args.dx,
        dy=args.dy,
        device=args.device,
    )

    _dump(ds_pair, args.out_pair)
    _dump(clf_ds, args.out_classifier)
    print(
        f"saved shared pair -> {args.out_pair} "
        f"(target timepoint={ds_pair.timepoints_ordered[-1]}, n_pcs={ds_pair.X_pca.shape[1]})",
    )
    print(
        f"saved classifier slide -> {args.out_classifier} "
        f"({len(clf_ds.ct)} cells, {len(clf_ds.ct_ordered)} classes)",
    )


if __name__ == "__main__":
    _main()
