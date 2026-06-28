"""RETIRED — raw-gene-PCA classifier-training slide prep (commented out below for reference).

This built a neutral classifier slide in the *raw-gene PCA + raw coordinates* space for the
standalone ``evaluate`` / ``otcfm`` path. That representation produces classifier metrics that are
**not comparable** to the NicheFlow models, so it has been retired in favour of the unified flow.

For the unified cross-model comparison (see ``docs/comparability_plan.md``):

1. build the *one* shared classifier slide with
   :func:`~paired_slides_eval.adapters.nicheflow.preprocess.preprocess_classifier_slide`
   (NicheFlow recipe: normalize_total+log1p shared PCA + standardised coords);
2. train one ``SpatialCTClassifierNet`` on it at a fixed ``n_neighbors``;
3. evaluate every model against the same shared pickle with ``evaluate --shared_pca``.

The original implementation is preserved, commented out, at the bottom of this file.
"""

from __future__ import annotations

_RETIRED_MSG = (
    "prepare_classifier_slide is retired (raw-gene-PCA path). Build the shared classifier slide "
    "with paired_slides_eval.adapters.nicheflow.preprocess.preprocess_classifier_slide and evaluate "
    "every model against the same shared pickle via `evaluate --shared_pca`. "
    "See docs/comparability_plan.md."
)


def prepare_classifier_slide(*_args, **_kwargs):
    """RETIRED — see the module docstring; use ``preprocess_classifier_slide`` + ``--shared_pca``."""
    raise NotImplementedError(_RETIRED_MSG)


def _main() -> None:
    raise SystemExit(_RETIRED_MSG)


if __name__ == "__main__":
    _main()


# --- RETIRED implementation (raw-gene-PCA path), preserved for reference -----------------------
# def prepare_classifier_slide(
#     target_h5ad, classifier_slide_h5ad, *,
#     ct_key="class", n_pcs=50, spatial_key="spatial",
#     radius=0.15, dx=0.15, dy=0.2, device="cpu",
# ):
#     from paired_slides_eval.adapters.nicheflow.preprocess import (
#         preprocess_classifier_slide_into_pca,
#     )
#     from paired_slides_eval.contract import TargetSlide
#     from paired_slides_eval.data.anndata import cell_type_labels, read_anndata
#     # Fit the target exactly as evaluate.py does: PCA on the target's raw genes + its label order.
#     target = TargetSlide.from_anndata(
#         target_h5ad, ct_key=ct_key, spatial_key=spatial_key, n_pcs=n_pcs
#     )
#     if target.pca is None:
#         raise ValueError("Pass n_pcs so the target PCA exists; the classifier trains in it.")
#     target_adata = read_anndata(target_h5ad)
#     _, ct_to_int = cell_type_labels(target_adata, ct_key)
#     if ct_to_int is None:
#         raise ValueError(f"ct_key {ct_key!r} not found on the target; needed for the classifier.")
#     return preprocess_classifier_slide_into_pca(
#         classifier_slide_h5ad, target.pca, target_adata.var_names,
#         list(ct_to_int),  # category order == eval's label mapping
#         cell_type_column=ct_key, radius=radius, dx=dx, dy=dy, device=device,
#     )
#
#
# def _main() -> None:
#     import argparse
#     import pickle
#     from pathlib import Path
#     ap = argparse.ArgumentParser(
#         description="Prepare a classifier-training slide .pkl for the Hydra classifier trainer."
#     )
#     ap.add_argument("--target", required=True, help="target slide .h5ad (defines PCA + labels)")
#     ap.add_argument("--classifier_slide", required=True,
#                     help="held-out slide .h5ad to train on (close but different from the target)")
#     ap.add_argument("--ct_key", default="class", help="obs column with cell types")
#     ap.add_argument("--n_pcs", type=int, default=50, help="PCA components (must match evaluate)")
#     ap.add_argument("--spatial_key", default="spatial")
#     ap.add_argument("--radius", type=float, default=0.15)
#     ap.add_argument("--dx", type=float, default=0.15)
#     ap.add_argument("--dy", type=float, default=0.2)
#     ap.add_argument("--device", default="cpu")
#     ap.add_argument("--out", required=True, help="path to write the .pkl (the trainer's data_fp)")
#     args = ap.parse_args()
#     ds = prepare_classifier_slide(
#         args.target, args.classifier_slide, ct_key=args.ct_key, n_pcs=args.n_pcs,
#         spatial_key=args.spatial_key, radius=args.radius, dx=args.dx, dy=args.dy, device=args.device,
#     )
#     out = Path(args.out)
#     out.parent.mkdir(parents=True, exist_ok=True)
#     with out.open("wb") as fh:
#         pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
#     n_cells = len(ds.ct)
#     print(f"saved classifier-slide dataclass -> {out} "
#           f"({n_cells} cells, {len(ds.ct_ordered)} classes, n_pcs={ds.X_pca.shape[1]})")
