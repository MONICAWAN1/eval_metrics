"""The NicheFlow implementation of the :class:`~nicheflow_eval.pipeline.run.Generator` contract.

``nicheflow_generator`` is a drop-in ``generator=`` for
:func:`nicheflow_eval.pipeline.run.run_pipeline`. It does the model-specific work the generic
pipeline deliberately knows nothing about:

1. preprocess the source+target slides into NicheFlow's niche dataclass (shared PCA, per-slide
   standardized coordinates, radius graph, grid subsample);
2. build the comparable ``TargetSlide`` in the same standardized ``X_pca`` space the model uses;
3. sample the flow to generate the target niches;
4. optionally train / load the spatial classifier for the ``ct/*`` groups.

Everything here imports ``nicheflow`` and/or ``torch``/``scanpy``; the rest of the package does
not. Needs the ``[pipeline]`` extra (``pip install -e ../nicheflow_mba``).
"""

from __future__ import annotations

import numpy as np

from nicheflow_eval.adapters.nicheflow.generate import generate
from nicheflow_eval.adapters.nicheflow.preprocess import (
    preprocess_classifier_slide,
    preprocess_pair,
)
from nicheflow_eval.contract import TargetSlide
from nicheflow_eval.pipeline.run import GenerationOutput


def target_from_dataclass(ds, timepoint: str, n_pcs: int | None = None) -> TargetSlide:
    """Build a ``TargetSlide`` from a preprocessed niche dataclass and a timepoint.

    The generated cells live in this dataclass's **standardized ``X_pca``** space, so the real
    target must come from the same dataclass (not raw genes) for the spaces to match. Slices the
    timepoint's cells and maps cell-type labels to ints.
    """
    cells = np.asarray(ds.timepoint_indices[timepoint])
    x = np.asarray(ds.X_pca[cells])
    if n_pcs is not None:
        x = x[:, :n_pcs]
    pos = np.asarray(ds.coords[cells])

    ct_raw = np.asarray(ds.ct)[cells]
    if np.issubdtype(ct_raw.dtype, np.integer):
        ct = ct_raw.astype(np.int64)
    else:
        ct = np.array([ds.ct_to_int[c] for c in ct_raw], dtype=np.int64)

    return TargetSlide(x=x, pos=pos, ct=ct, n_classes=len(ds.ct_to_int), pca=None)


def nicheflow_generator(
    *,
    source,
    target,
    checkpoint: str,
    classifier_h5ad=None,
    classifier_ckpt: str | None = None,
    n_pcs: int = 50,
    cell_type_column: str = "class",
    radius: float = 0.15,
    dx: float = 0.15,
    dy: float = 0.2,
    device: str = "cpu",
    num_steps: int = 20,
    solver: str = "euler",
    variant: str = "cfm",
    n_slices: int | None = None,
    generated_out=None,
    classifier_train_kwargs: dict | None = None,
) -> GenerationOutput:
    """Preprocess -> generate -> (train/load classifier) for NicheFlow.

    Returns a ``GenerationOutput`` ready for :func:`nicheflow_eval.pipeline.run.run_pipeline`.

    Args:
        source / target: source and target slides (raw genes + ``obsm['spatial']``; AnnData/path).
        checkpoint: trained flow checkpoint.
        classifier_h5ad: held-out slide to train the spatial classifier on (enables ``ct/*``).
            Projected into the source+target PCA basis + label space.
        classifier_ckpt: load a pre-trained spatial classifier instead of training one.
        n_pcs: PCs for the shared PCA. cell_type_column: ``obs`` column with cell types.
        radius/dx/dy: niche radius graph + grid-subsample resolution.
        num_steps/solver/variant/n_slices: flow sampler settings (must match the checkpoint).
        generated_out: optional path to write the generated cells as ``.h5ad``.
    """
    ds_pair, pre = preprocess_pair(
        source,
        target,
        n_pcs=n_pcs,
        cell_type_column=cell_type_column,
        radius=radius,
        dx=dx,
        dy=dy,
        device=device,
    )
    target_tp = ds_pair.timepoints_ordered[-1]
    target_slide = target_from_dataclass(ds_pair, target_tp)

    gen_result = generate(
        ds_pair,
        checkpoint,
        target_timepoint=target_tp,
        n_slices=n_slices,
        num_steps=num_steps,
        solver=solver,
        variant=variant,
        device=device,
    )
    if generated_out is not None:
        gen_result.to_anndata().write_h5ad(str(generated_out))

    classifier = _resolve_classifier(
        classifier_ckpt=classifier_ckpt,
        classifier_h5ad=classifier_h5ad,
        base_preprocessor=pre,
        n_pcs=target_slide.x.shape[1],
        n_classes=target_slide.n_classes,
        cell_type_column=cell_type_column,
        radius=radius,
        dx=dx,
        dy=dy,
        device=device,
        train_kwargs=classifier_train_kwargs or {},
    )

    return GenerationOutput(
        target=target_slide, generated=gen_result.to_generated_niches(), classifier=classifier
    )


def _resolve_classifier(
    *,
    classifier_ckpt,
    classifier_h5ad,
    base_preprocessor,
    n_pcs,
    n_classes,
    cell_type_column,
    radius,
    dx,
    dy,
    device,
    train_kwargs,
):
    """Load a classifier checkpoint, or train one on the projected classifier slide, or ``None``."""
    if classifier_ckpt is not None:
        from nicheflow_eval.evaluate import build_spatial_classifier

        return build_spatial_classifier(classifier_ckpt, n_pcs, n_classes)

    if classifier_h5ad is not None:
        from nicheflow_eval.classifier.train_helper import train_spatial_classifier

        clf_ds = preprocess_classifier_slide(
            classifier_h5ad,
            base_preprocessor,
            cell_type_column=cell_type_column,
            radius=radius,
            dx=dx,
            dy=dy,
            device=device,
        )
        return train_spatial_classifier(clf_ds, n_pcs=n_pcs, n_classes=n_classes, **train_kwargs)

    return None
