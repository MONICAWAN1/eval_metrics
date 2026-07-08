"""The NicheFlow implementation of the :class:`~paired_slides_eval.pipeline.run.Generator` contract.

``nicheflow_generator`` is a drop-in ``generator=`` for
:func:`paired_slides_eval.pipeline.run.run_pipeline`. It does the model-specific work the generic
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

from paired_slides_eval.adapters.base import BaseGenerator
from paired_slides_eval.adapters.nicheflow.generate import generate
from paired_slides_eval.adapters.nicheflow.preprocess import (
    preprocess_classifier_slide,
    preprocess_pair,
)
from paired_slides_eval.contract import TargetSlide
from paired_slides_eval.pipeline.run import GenerationOutput


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


class NicheFlowGenerator(BaseGenerator):
    """NicheFlow generation adapter.

    Preprocesses the source+target pair into NicheFlow's niche scaffolding (shared PCA, per-slide
    coordinate standardization, radius graph + grid subsample), samples the flow to produce the
    target niches, and optionally trains or loads the spatial classifier for the ``ct/*`` groups.
    The generated cells and the target both live in the standardized ``X_pca`` space, so they are
    directly comparable.

    Construction parameters (supplied by ``configs/generator/nicheflow.yaml``):
        n_pcs: components for the shared PCA.
        cell_type_column: ``obs`` column holding cell-type labels.
        radius / dx / dy: radius-graph and grid-subsample resolution.
        device: torch device for preprocessing, sampling, and training.
        num_steps / solver / variant / n_slices: flow sampler settings (must match the checkpoint).
        classifier_h5ad: held-out slide to train the spatial classifier on (enables ``ct/*``).
        classifier_ckpt: load a pre-trained spatial classifier instead of training one.
        classifier_train_kwargs: extra keyword arguments for classifier training.
    """

    def __init__(
        self,
        *,
        n_pcs: int = 50,
        cell_type_column: str = "class",
        radius: float = 0.15,
        dx: float = 0.15,
        dy: float = 0.2,
        device: str = "cpu",
        num_steps: int = 20,
        solver: str = "euler",
        variant: str = "cfm",
        vfm_objective: str = "GLVFM",
        n_slices: int | None = None,
        classifier_h5ad=None,
        classifier_ckpt: str | None = None,
        classifier_train_kwargs: dict | None = None,
    ) -> None:
        self.n_pcs = n_pcs
        self.cell_type_column = cell_type_column
        self.radius = radius
        self.dx = dx
        self.dy = dy
        self.device = device
        self.num_steps = num_steps
        self.solver = solver
        self.variant = variant
        self.vfm_objective = vfm_objective
        self.n_slices = n_slices
        self.classifier_h5ad = classifier_h5ad
        self.classifier_ckpt = classifier_ckpt
        self.classifier_train_kwargs = classifier_train_kwargs or {}

    def __call__(self, *, source, target, checkpoint, **_) -> GenerationOutput:
        ds_pair, pre = preprocess_pair(
            source,
            target,
            n_pcs=self.n_pcs,
            cell_type_column=self.cell_type_column,
            radius=self.radius,
            dx=self.dx,
            dy=self.dy,
            device=self.device,
        )
        target_tp = ds_pair.timepoints_ordered[-1]
        target_slide = target_from_dataclass(ds_pair, target_tp)

        gen_result = generate(
            ds_pair,
            checkpoint,
            target_timepoint=target_tp,
            n_slices=self.n_slices,
            num_steps=self.num_steps,
            solver=self.solver,
            variant=self.variant,
            vfm_objective=self.vfm_objective,
            device=self.device,
        )

        classifier = _resolve_classifier(
            classifier_ckpt=self.classifier_ckpt,
            classifier_h5ad=self.classifier_h5ad,
            base_preprocessor=pre,
            n_pcs=target_slide.x.shape[1],
            n_classes=target_slide.n_classes,
            cell_type_column=self.cell_type_column,
            radius=self.radius,
            dx=self.dx,
            dy=self.dy,
            device=self.device,
            train_kwargs=self.classifier_train_kwargs,
        )

        return GenerationOutput(
            target=target_slide, generated=gen_result.to_generated_niches(), classifier=classifier
        )


def nicheflow_generator(*, source, target, checkpoint, **kwargs) -> GenerationOutput:
    """Functional wrapper around :class:`NicheFlowGenerator` (kept for the Python API)."""
    return NicheFlowGenerator(**kwargs)(source=source, target=target, checkpoint=checkpoint)


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
        from paired_slides_eval.probes import build_spatial_classifier

        return build_spatial_classifier(classifier_ckpt, n_pcs, n_classes)

    if classifier_h5ad is not None:
        from paired_slides_eval.classifier.train_helper import train_spatial_classifier

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
