"""Export the shared source+target PCA (NicheFlow recipe) as an ``fm_mnist``
``load_spatial_pca`` stats ``.npz``, so OT-CFM can be **trained in that exact
basis** instead of fitting its own.

The OT-CFM trainer (``fm_mnist/scripts/train_cfm_spatial.py --pca_stats <npz>``) reads this file and
projects its slide through the injected basis (see ``fm.data.load_spatial_pca(external_stats=...)``),
so the generated cells land directly in the shared whitened-PCA(50) space the niche models live in --
no post-hoc inversion/reprojection, and no rank deficiency. This keeps ``fm_mnist`` decoupled: it only
reads a plain ``.npz`` and never imports ``paired_slides_eval``/``nicheflow``.

The npz keys mirror what ``load_spatial_pca`` stores in its own ``stats`` dict, so
``invert_pca_expression`` and the eval both see the shared basis:
``pca_components (k, G)``, ``pca_mean (G,)``, ``sc_mean (k,)``, ``sc_scale (k,)``, ``target_sum``,
``var_names``. It also carries the target slide's **coordinate** standardiser
(``coord_mean``/``coord_std``, the per-slide all-cells stats NicheFlow uses), so OT-CFM trains on —
and emits — coordinates in the *same* standardised frame the niche models live in (symmetric with the
expression fix): no train-split z-score, no raw round-trip, and the eval passes the coords through.

Usage::

    python -m paired_slides_eval.adapters.otcfm_export --pair data/abca_pair.pkl --out shared_pca.npz

"""

from __future__ import annotations

import numpy as np


def export_shared_pca(pair_pkl: str, out_npz: str) -> dict:
    """Read a preprocessed pair ``.pkl`` and write its shared basis as an
    ``fm_mnist`` stats ``.npz``.

    A thin front end over :meth:`~paired_slides_eval.data.shared_pca.Basis.to_fm_npz`, which owns the
    ``SharedGenePCA``/coord → ``fm_mnist`` mapping so the training file and the eval projection are
    guaranteed to describe the same basis.

    """
    from paired_slides_eval.data.dataclass import load_h5ad_dataset_dataclass
    from paired_slides_eval.data.shared_pca import Basis

    ds = load_h5ad_dataset_dataclass(pair_pkl)
    return Basis.from_dataclass(ds).to_fm_npz(out_npz)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Export the shared PCA as an fm_mnist load_spatial_pca .npz (for OT-CFM training).",
    )
    ap.add_argument("--pair", required=True, help="preprocessed pair .pkl (from preprocess_pair)")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()

    s = export_shared_pca(args.pair, args.out)
    print(
        f"wrote {args.out}: k={s['n_pcs']} PCs, {len(s['var_names'])} genes, "
        f"target_sum={float(s['target_sum']):.1f}, "
        f"coord_std={np.round(s['coord_std'], 2).tolist()}",
    )


if __name__ == "__main__":
    _main()
