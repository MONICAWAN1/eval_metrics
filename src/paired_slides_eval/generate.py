"""Model-agnostic generation entry point: run any generator and write the cells to disk.

This decouples **generation** from **evaluation**. :func:`paired_slides_eval.pipeline.run_pipeline`
does generate+evaluate in one shot; this module does *only* generation, so you can generate once,
persist the cells, and evaluate later (or repeatedly) with ``python -m paired_slides_eval.evaluate``.

The generator is the same blackbox contract the pipeline uses
(:class:`paired_slides_eval.pipeline.run.Generator`): any callable
``(*, source, target, checkpoint, **kwargs) -> GenerationOutput``. This module never imports
``torch``/``nicheflow``/``scanpy`` itself — it only *orchestrates* (resolve the generator, call it,
serialize the result); the model-specific imports live inside whichever generator you pick.

From the CLI you select a generator by dotted path (``module.path:callable``) — this is required,
the package favours no model. The bundled NicheFlow adapter
(``paired_slides_eval.adapters.nicheflow:nicheflow_generator``) is one choice::

    python -m paired_slides_eval.generate \\
        --generator mypkg.mymodel:my_generator \\
        --source SRC.h5ad --target TGT.h5ad --checkpoint M.ckpt \\
        --generated_out gen.h5ad --gen-kwarg n_pcs=50 --gen-kwarg radius=0.15

The written file round-trips through the evaluator's loader — the same ``.h5ad`` / ``.npz`` layouts
:func:`paired_slides_eval.evaluate._load_generated` reads — so the two steps compose cleanly::

    python -m paired_slides_eval.evaluate --target TGT.h5ad --generated gen.h5ad
"""

from __future__ import annotations

import numpy as np

from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide
from paired_slides_eval.pipeline.run import GenerationOutput


def resolve_generator(spec):
    """Resolve a generator from a ``"module.path:callable"`` spec (or pass a callable through).

    The dotted-path form lets a shell user point the CLI at *their own* model without any
    packaging — e.g. ``mypkg.mymodel:my_generator``. A callable is returned unchanged so the
    Python API can hand in a function directly.
    """
    if callable(spec):
        return spec
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(
            f"Generator spec must be 'module.path:callable' (e.g. "
            f"'paired_slides_eval.adapters.nicheflow:nicheflow_generator'); got {spec!r}."
        )
    import importlib

    module_path, _, attr = spec.partition(":")
    module = importlib.import_module(module_path)
    try:
        generator = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"module {module_path!r} has no attribute {attr!r}.") from exc
    if not callable(generator):
        raise TypeError(f"{spec!r} resolved to a non-callable {type(generator).__name__}.")
    return generator


def write_generated(generated: GeneratedNiches | GeneratedSlide, path: str) -> str:
    """Serialize generated cells to ``path`` (``.h5ad`` or ``.npz``), symmetric with the loader.

    The layouts match :func:`paired_slides_eval.evaluate._load_generated` /
    :meth:`GeneratedNiches.from_anndata`:

    * ``.h5ad`` — niche-shaped cells become flat rows with ``obs['niche_id']`` grouping each niche
      (centroid first), coords in ``obsm['spatial']``, and any paired ground truth in
      ``obsm['gt_x']`` / ``obsm['gt_pos']`` / ``obs['gt_ct']``; a flat slide is ``X`` +
      ``obsm['spatial']``.
    * ``.npz`` — niche-shaped: 3-D ``x`` / ``pos`` (+ optional ``gt_x`` / ``gt_pos`` / ``gt_ct``);
      flat: 2-D ``x`` / ``pos``.

    Returns ``path``.
    """
    path = str(path)
    if path.endswith(".npz"):
        _write_npz(generated, path)
    elif path.endswith(".h5ad"):
        _write_h5ad(generated, path)
    else:
        raise ValueError(f"Unsupported output extension for {path!r}; use '.h5ad' or '.npz'.")
    return path


def _write_npz(generated: GeneratedNiches | GeneratedSlide, path: str) -> None:
    arrays = {"x": np.asarray(generated.x), "pos": np.asarray(generated.pos)}
    if isinstance(generated, GeneratedNiches):
        for key in ("gt_x", "gt_pos", "gt_ct"):
            value = getattr(generated, key, None)
            if value is not None:
                arrays[key] = np.asarray(value)
    np.savez(path, **arrays)


def _write_h5ad(generated: GeneratedNiches | GeneratedSlide, path: str) -> None:
    import anndata as ad
    import pandas as pd

    if isinstance(generated, GeneratedNiches):
        b, n, _ = generated.x.shape
        adata = ad.AnnData(X=generated.x.reshape(b * n, -1).astype(np.float32))
        adata.obs["niche_id"] = np.repeat(np.arange(b), n)
        adata.obsm["spatial"] = generated.pos.reshape(b * n, -1).astype(np.float32)
        if generated.gt_x is not None and generated.gt_pos is not None:
            adata.obsm["gt_x"] = generated.gt_x.reshape(b * n, -1).astype(np.float32)
            adata.obsm["gt_pos"] = generated.gt_pos.reshape(b * n, -1).astype(np.float32)
        if generated.gt_ct is not None:
            adata.obs["gt_ct"] = np.repeat(np.asarray(generated.gt_ct), n)
            adata.obs = adata.obs.astype({"gt_ct": "int64"})
        adata.obs = adata.obs.astype({"niche_id": "int64"})
        adata.obs.index = pd.RangeIndex(b * n).astype(str)
    else:  # GeneratedSlide
        adata = ad.AnnData(X=np.asarray(generated.x, dtype=np.float32))
        adata.obsm["spatial"] = np.asarray(generated.pos, dtype=np.float32)
        adata.obs.index = pd.RangeIndex(generated.x.shape[0]).astype(str)
    adata.write_h5ad(path)


def generate_cells(
    source,
    target,
    checkpoint: str,
    *,
    generator,
    out: str | None = None,
    **generator_kwargs,
) -> GenerationOutput:
    """Run a generator to produce cells and, if ``out`` is given, write them to disk.

    Args:
        source / target: raw slides (AnnData or ``.h5ad`` paths), passed straight to the generator.
        checkpoint: the trained model checkpoint, passed straight to the generator.
        generator: a :class:`~paired_slides_eval.pipeline.run.Generator` callable, or a
            ``"module.path:callable"`` spec resolved via :func:`resolve_generator` (required — the
            bundled NicheFlow adapter is one choice).
        out: optional path (``.h5ad`` / ``.npz``) to write the generated cells to.
        **generator_kwargs: any extra options the chosen generator accepts (e.g. the NicheFlow
            adapter's ``n_pcs``, ``radius``, ``num_steps`` …).

    Returns the full :class:`~paired_slides_eval.pipeline.run.GenerationOutput` (target +
    generated, and a ``classifier`` if the generator trained one). Only the **generated cells** are
    persisted; evaluate them later with :mod:`paired_slides_eval.evaluate`.
    """
    generator = resolve_generator(generator)
    output = generator(source=source, target=target, checkpoint=checkpoint, **generator_kwargs)
    if out is not None:
        write_generated(output.generated, out)
    return output


def _coerce(value: str):
    """Best-effort scalar coercion for ``--gen-kwarg`` values (none/bool/int/float, else str)."""
    low = value.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate cells with any model (the blackbox generator) and write them to disk "
        "— no evaluation. Evaluate the output later with `python -m paired_slides_eval.evaluate`."
    )
    ap.add_argument(
        "--generator",
        required=True,
        help="generator as 'module.path:callable' (e.g. the bundled "
        "paired_slides_eval.adapters.nicheflow:nicheflow_generator)",
    )
    ap.add_argument("--source", required=True, help="source slide .h5ad (raw genes + coords)")
    ap.add_argument("--target", required=True, help="target slide .h5ad to generate")
    ap.add_argument("--checkpoint", required=True, help="trained model checkpoint (-> generator)")
    ap.add_argument(
        "--generated_out",
        required=True,
        help="where to write the generated cells (.h5ad or .npz)",
    )
    ap.add_argument(
        "--gen-kwarg",
        action="append",
        default=[],
        dest="gen_kwargs",
        metavar="KEY=VALUE",
        help="extra generator option, repeatable (e.g. --gen-kwarg n_pcs=50 --gen-kwarg "
        "radius=0.15). Values are coerced to none/bool/int/float when possible, else kept as text.",
    )
    args = ap.parse_args()

    kwargs = {}
    for item in args.gen_kwargs:
        if "=" not in item:
            ap.error(f"--gen-kwarg expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        kwargs[key] = _coerce(value)

    output = generate_cells(
        args.source,
        args.target,
        args.checkpoint,
        generator=args.generator,
        out=args.generated_out,
        **kwargs,
    )

    g = output.generated
    if isinstance(g, GeneratedNiches):
        shape = f"{g.x.shape[0]} niches x {g.x.shape[1]} points"
    else:
        shape = f"{g.x.shape[0]} cells, {g.x.shape[1]} feats (flat slide)"
    print(f"generated: {shape} -> {args.generated_out}")


# Usage: python -m paired_slides_eval.generate --source S.h5ad --target T.h5ad \
#        --checkpoint M.ckpt --generated_out gen.h5ad
if __name__ == "__main__":
    _main()
