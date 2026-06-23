"""End-to-end pipeline: raw AnnData slides + a checkpoint -> generated cells -> metrics.

``run_pipeline`` is the user-friendly entry point: give it the **source** and **target** slides
(``.h5ad``, raw genes + coords), a slide to **train the classifier** on, and a trained flow
**checkpoint**. It preprocesses the niches, generates the target with the flow (blackbox import of
``nicheflow``), trains/loads the classifier, and runs the full metric suite — everything in the
preprocessor's shared, standardized space so the pieces are comparable.

For evaluating cells you generated elsewhere, skip this and use
:func:`nicheflow_eval.evaluate.evaluate` directly on AnnData (the standalone path).

Needs the ``nicheflow`` package (the ``[pipeline]`` extra) for the generation step.
"""

from __future__ import annotations

from dataclasses import dataclass

from nicheflow_eval.contract import GeneratedNiches, TargetSlide
from nicheflow_eval.evaluate import ALL_GROUPS, evaluate
from nicheflow_eval.pipeline.generate import GenerationResult, generate
from nicheflow_eval.preprocessing import preprocess_classifier_slide, preprocess_pair


@dataclass
class PipelineResult:
    metrics: dict
    target: TargetSlide
    generated: GeneratedNiches
    generation: GenerationResult


def run_pipeline(
    source_h5ad,
    target_h5ad,
    checkpoint: str,
    *,
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
    groups: tuple[str, ...] = ALL_GROUPS,
    seed: int = 0,
    generated_out=None,
    classifier_train_kwargs: dict | None = None,
) -> PipelineResult:
    """Preprocess -> generate -> (train/load classifier) -> evaluate. Returns a ``PipelineResult``.

    Args:
        source_h5ad / target_h5ad: the source and target slides (raw genes + ``obsm['spatial']``).
        checkpoint: trained flow checkpoint.
        classifier_h5ad: held-out slide to train the neutral classifier on (enables the ``ct/*``
            groups). Projected into the source+target PCA basis + label space.
        classifier_ckpt: load a pre-trained spatial classifier instead of training one.
        n_pcs: PCs for the shared PCA. cell_type_column: ``obs`` column with cell types.
        radius/dx/dy: niche radius graph + grid-subsample resolution (preprocessing).
        num_steps/solver/variant/n_slices: flow sampler settings (must match the checkpoint).
        generated_out: optional path to write the generated cells as ``.h5ad``.
    """
    ds_pair, pre = preprocess_pair(
        source_h5ad,
        target_h5ad,
        n_pcs=n_pcs,
        cell_type_column=cell_type_column,
        radius=radius,
        dx=dx,
        dy=dy,
        device=device,
    )
    target_tp = ds_pair.timepoints_ordered[-1]
    target = TargetSlide.from_dataclass(ds_pair, timepoint=target_tp)

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
    generated = gen_result.to_generated_niches()

    if generated_out is not None:
        gen_result.to_anndata().write_h5ad(str(generated_out))

    classifier = _resolve_classifier(
        classifier_ckpt=classifier_ckpt,
        classifier_h5ad=classifier_h5ad,
        base_preprocessor=pre,
        n_pcs=target.x.shape[1],
        n_classes=target.n_classes,
        cell_type_column=cell_type_column,
        radius=radius,
        dx=dx,
        dy=dy,
        device=device,
        train_kwargs=classifier_train_kwargs or {},
    )

    metrics = evaluate(target, generated, classifier=classifier, groups=groups, seed=seed)
    return PipelineResult(
        metrics=metrics, target=target, generated=generated, generation=gen_result
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
