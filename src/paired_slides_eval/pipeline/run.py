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
    :class:`GeneratedSlide` (``X`` + ``obsm['spatial']``). With a flat slide the niche metrics
    (regression, concordance, ct_gap) are skipped.

    Args:
        generated_adata_or_path: the generated cells (AnnData or path).
        target_adata_or_path: the raw target slide (AnnData or path).
        ct_key: ``obs`` column with cell types on the target (enables the ``ct/*`` groups).
        n_pcs: PCs to fit on the target and project both sides into (``None`` keeps raw genes).
        niche_key: ``obs`` column marking the niche layout (default ``"niche_id"``).
        target_kwargs / generated_kwargs: extra kwargs forwarded to the respective ``from_anndata``.
    """
    from paired_slides_eval.data.anndata import read_anndata

    target = TargetSlide.from_anndata(
        target_adata_or_path, ct_key=ct_key, n_pcs=n_pcs, **(target_kwargs or {})
    )
    gen_adata = read_anndata(generated_adata_or_path)
    if niche_key in gen_adata.obs:
        generated = GeneratedNiches.from_anndata(
            gen_adata, niche_key=niche_key, **(generated_kwargs or {})
        )
    else:
        generated = GeneratedSlide.from_anndata(gen_adata, **(generated_kwargs or {}))
    return GenerationOutput(target=target, generated=generated.project(target.pca))


def run_pipeline(
    source,
    target,
    checkpoint: str,
    *,
    generator: Generator,
    classifier=None,
    groups: tuple[str, ...] = ALL_GROUPS,
    seed: int = 0,
    **generator_kwargs,
) -> PipelineResult:
    """Generate with ``generator`` and run the metric suite. Returns a :class:`PipelineResult`.

    Args:
        source / target: raw slides (AnnData or ``.h5ad`` paths), passed straight to ``generator``.
        checkpoint: the trained model checkpoint, passed straight to ``generator``.
        generator: any callable implementing the :class:`Generator` contract (e.g.
            ``paired_slides_eval.adapters.nicheflow.nicheflow_generator``).
        classifier: optional fallback for the ``ct/*`` groups when the generator does not return
            one — either a ready classifier ``nn.Module`` or a path to a ``.ckpt``.
        groups / seed: forwarded to :func:`paired_slides_eval.evaluate.evaluate`.
        **generator_kwargs: any extra options your ``generator`` accepts (e.g. the NicheFlow
            adapter's ``n_pcs``, ``radius``, ``classifier_h5ad`` …).
    """
    out = generator(source=source, target=target, checkpoint=checkpoint, **generator_kwargs)

    clf = out.classifier
    if clf is None and classifier is not None:
        clf = _resolve_classifier_arg(classifier, out.target)

    metrics = evaluate(out.target, out.generated, classifier=clf, groups=groups, seed=seed)
    return PipelineResult(metrics=metrics, target=out.target, generated=out.generated)


def _resolve_classifier_arg(classifier, target: TargetSlide):
    """Accept a ready classifier module as-is, or load one from a ``.ckpt`` path."""
    if isinstance(classifier, str):
        from paired_slides_eval.evaluate import build_spatial_classifier

        return build_spatial_classifier(classifier, target.x.shape[1], target.n_classes)
    return classifier
