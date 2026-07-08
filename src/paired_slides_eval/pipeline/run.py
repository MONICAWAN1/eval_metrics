"""Model-agnostic pipeline: raw AnnData slides + a checkpoint -> generated cells -> metrics.

This layer knows **nothing** about any particular generative model. You hand it a ``generator``
— any callable that turns the raw slides + a checkpoint into a comparable
``(TargetSlide, GeneratedNiches)`` pair — and it runs the metric suite on the result.

Two ways to use it:

* Use your own model: Write a function matching the :class:`Generator` protocol (see its
  docstring for the one-method contract) and pass it as ``generator=...``. Your function does the
  generation however it likes; :func:`from_generated_anndata` turns its output (a generated
  ``.h5ad`` in gene space) into the pair this pipeline expects, in ~1 line.
* Use the bundled NicheFlow adapter: ``from paired_slides_eval.adapters.nicheflow import
  nicheflow_generator`` and pass it as ``generator=nicheflow_generator`` (needs the ``[pipeline]``
  extra). It is just one implementation of :class:`Generator`.

For cells you generated entirely elsewhere, you do not need this module at all — call
:func:`paired_slides_eval.evaluate.evaluate` directly on a ``TargetSlide`` + ``GeneratedNiches``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide, TargetSlide
from paired_slides_eval.evaluate import ALL_GROUPS, evaluate


@dataclass
class GenerationOutput:
    """What a :class:`Generator` returns: a ready-to-evaluate (target, generated) pair.

    The generator owns the one genuinely model-specific job — making the real target and the
    generated cells live in the **same feature space** (see :func:`from_generated_anndata` for the
    common gene-space recipe). Optionally it can also hand back a trained ``classifier`` for the
    ``ct/*`` metric groups; if it does not, pass a checkpoint/object via ``run_pipeline(...,
    classifier=...)`` instead.
    """

    target: TargetSlide
    generated: GeneratedNiches | GeneratedSlide
    classifier: object | None = None


@runtime_checkable
class Generator(Protocol):
    """The blackbox generation contract — implement this for any model.

    A generator is any callable with this signature::

        def my_generator(*, source, target, checkpoint, **kwargs) -> GenerationOutput:
            ...

    where ``source`` / ``target`` are raw AnnData slides (or paths to ``.h5ad``) and ``checkpoint``
    is your trained model. Return a :class:`GenerationOutput`; the easiest way is to generate cells
    into an AnnData (gene space) and call :func:`from_generated_anndata`.
    """

    def __call__(self, *, source, target, checkpoint: str, **kwargs) -> GenerationOutput: ...


@dataclass
class PipelineResult:
    metrics: dict
    target: TargetSlide
    generated: GeneratedNiches | GeneratedSlide


def from_generated_anndata(
    generated_adata_or_path,
    target_adata_or_path,
    *,
    ct_key: str | None = None,
    n_pcs: int | None = 50,
    niche_key: str = "niche_id",
    target_kwargs: dict | None = None,
    generated_kwargs: dict | None = None,
) -> GenerationOutput:
    """Build a :class:`GenerationOutput` from a generated ``.h5ad`` and the raw target slide.

    The common path for a bring-your-own-model generator: your model writes generated cells as an
    AnnData in **gene space** (same genes as the target). This fits one PCA on the target and
    projects the generated cells through it, so both sides share a basis.

    The generated layout is auto-detected: niche-shaped (:class:`GeneratedNiches`) if
    ``obs[niche_key]`` is present, otherwise a flat whole-slide
    :class:`GeneratedSlide` (``X`` + ``obsm['spatial']``). For a flat slide the classifier metrics
    (concordance, ct_gap) are still computed by reconstructing niches from geometry; only
    regression (needs matched ground truth) is skipped.

    Args:
        generated_adata_or_path: the generated cells (AnnData or path).
        target_adata_or_path: the raw target slide (AnnData or path).
        ct_key: ``obs`` column with cell types on the target (enables the ``ct/*`` groups).
        n_pcs: PCs to fit on the target and project both sides into (``None`` keeps raw genes).
        niche_key: ``obs`` column marking the niche layout (default ``"niche_id"``).
        target_kwargs / generated_kwargs: extra kwargs forwarded to the respective ``from_anndata``.
    """
    from paired_slides_eval.data.anndata import read_anndata

    target = _resolve_target(target_adata_or_path, ct_key=ct_key, n_pcs=n_pcs, **(target_kwargs or {}))
    gen_adata = read_anndata(generated_adata_or_path)
    if niche_key in gen_adata.obs:
        generated = GeneratedNiches.from_anndata(
            gen_adata, niche_key=niche_key, **(generated_kwargs or {})
        )
    else:
        generated = GeneratedSlide.from_anndata(gen_adata, **(generated_kwargs or {}))
    return GenerationOutput(target=target, generated=generated.project(target.pca))


def _resolve_target(target, *, ct_key=None, n_pcs=50, **target_kwargs) -> TargetSlide:
    """Accept a ready :class:`TargetSlide` as-is, or build one from an AnnData / ``.h5ad`` path."""
    if isinstance(target, TargetSlide):
        return target
    return TargetSlide.from_anndata(target, ct_key=ct_key, n_pcs=n_pcs, **target_kwargs)


def from_generated_arrays(
    x,
    pos,
    target,
    *,
    gt_x=None,
    gt_pos=None,
    gt_ct=None,
    ct_key: str | None = None,
    n_pcs: int | None = 50,
    target_kwargs: dict | None = None,
) -> GenerationOutput:
    """Build a :class:`GenerationOutput` from in-memory generated arrays + a target.

    The array counterpart of :func:`from_generated_anndata`, for generators that return cells
    directly (no intermediate ``.h5ad``). Niche-shaped if ``x`` is 3-D ``(B, N, D)`` (optionally
    with ``gt_x``/``gt_pos``/``gt_ct``), else a flat ``(N, D)`` slide.

    Feature space is reconciled automatically (see
    :func:`~paired_slides_eval.contract._pca_aware_transform`): if the cells are in **gene space**
    they are projected through the target's PCA; if they are already **PCA-reduced** (a model that
    samples in latent space, like a flow), they are passed through unchanged. For the latter, supply
    the target already in that same basis — pass a ready ``TargetSlide`` (``pca=None``), or an
    AnnData with ``target_kwargs={"expr_key": "X_pca"}`` and ``n_pcs=None``.

    Args:
        x / pos: generated expression / coordinates, ``(B, N, D)`` (niche) or ``(N, D)`` (flat).
        target: a ready :class:`TargetSlide`, or an AnnData / ``.h5ad`` path to build one from.
        gt_x / gt_pos / gt_ct: optional paired ground truth (niche-shaped only).
        ct_key / n_pcs / target_kwargs: used only when building the target from AnnData.
    """
    import numpy as np

    target_slide = _resolve_target(target, ct_key=ct_key, n_pcs=n_pcs, **(target_kwargs or {}))
    x = np.asarray(x)
    if x.ndim == 3:
        generated = GeneratedNiches(x=x, pos=pos, gt_x=gt_x, gt_pos=gt_pos, gt_ct=gt_ct)
    else:
        generated = GeneratedSlide(x=x, pos=pos)
    return GenerationOutput(target=target_slide, generated=generated.project(target_slide.pca))


def run_pipeline(
    source,
    target,
    checkpoint: str,
    *,
    generator,
    classifier=None,
    groups: tuple[str, ...] = ALL_GROUPS,
    seed: int = 0,
    evaluate_kwargs: dict | None = None,
    **generator_kwargs,
) -> PipelineResult:
    """Generate with ``generator`` and run the metric suite. Returns a :class:`PipelineResult`.

    Args:
        source / target: raw slides (AnnData or ``.h5ad`` paths), passed straight to ``generator``.
        checkpoint: the trained model checkpoint, passed straight to ``generator``.
        generator: a :class:`Generator` — any callable (typically a :class:`BaseGenerator`
            instance, e.g. one built from a Hydra config) with the signature
            ``(*, source, target, checkpoint, **kwargs) -> GenerationOutput``.
        classifier: optional fallback for the ``ct/*`` groups when the generator does not return
            one — either a ready classifier ``nn.Module`` or a path to a ``.ckpt``.
        groups / seed: forwarded to :func:`paired_slides_eval.evaluate.evaluate`.
        evaluate_kwargs: extra keyword arguments forwarded to :func:`evaluate` (e.g.
            ``ct_real_reference="fixed"`` for cross-model-comparable ``ct/acc_real``). Kept separate
            from ``generator_kwargs`` so eval-only options do not leak into the generator call.
        **generator_kwargs: any extra options the ``generator`` accepts at call time.
    """
    out = generator(source=source, target=target, checkpoint=checkpoint, **generator_kwargs)

    clf = out.classifier
    if clf is None and classifier is not None:
        clf = _resolve_classifier_arg(classifier, out.target)

    metrics = evaluate(
        out.target, out.generated, classifier=clf, groups=groups, seed=seed,
        **(evaluate_kwargs or {}),
    )
    return PipelineResult(metrics=metrics, target=out.target, generated=out.generated)


def generate_cells(source, target, checkpoint, *, generator, out=None, **generator_kwargs):
    """Run ``generator`` to produce cells and, if ``out`` is given, write them to disk.

    The generate-only counterpart of :func:`run_pipeline` (no evaluation). ``generator`` is a
    :class:`Generator` callable / :class:`BaseGenerator` instance. Returns the full
    :class:`GenerationOutput`; only the generated cells are written.
    """
    output = generator(source=source, target=target, checkpoint=checkpoint, **generator_kwargs)
    if out is not None:
        from paired_slides_eval.pipeline.io import write_generated

        write_generated(output.generated, out)
    return output


def _resolve_classifier_arg(classifier, target: TargetSlide):
    """Accept a ready classifier module as-is, or load one from a ``.ckpt`` path."""
    if isinstance(classifier, str):
        from paired_slides_eval.probes import build_spatial_classifier

        return build_spatial_classifier(classifier, target.x.shape[1], target.n_classes)
    return classifier
