"""Run the metric suite and return a tidy ``pandas.DataFrame``.

A thin view over :func:`paired_slides_eval.evaluate.evaluate` /
:func:`~paired_slides_eval.evaluate.evaluate_files` — **the numbers are identical**, only reshaped
into a benchmarking table (metrics as rows, one value column per model). ``evaluate()`` stays the
kernel; this never re-runs anything or touches how cells are reconciled into the scored space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _to_frame(result: dict, name: str) -> "pd.DataFrame":
    """Reshape a flat ``{prefix/group/metric: value}`` dict into a one-column table."""
    import pandas as pd

    values = {k: v for k, v in result.items() if not k.startswith("_")}
    df = pd.DataFrame({name: pd.Series(values, dtype="float64")})
    df.index.name = "metric"
    df.attrs["skipped"] = list(result.get("_skipped", []))
    df.attrs["notes"] = list(result.get("_notes", []))
    return df


def metrics(target, generated, *, name: str = "value", **evaluate_kwargs) -> "pd.DataFrame":
    """Run the suite on an in-memory pair and return a tidy results **table**.

    Wraps :func:`paired_slides_eval.evaluate.evaluate`. The index is the flat metric key
    (``{prefix}/{group}/{metric}``); the single value column is named ``name`` (pass a distinct model
    name and ``pd.concat([...], axis=1)`` several to build a comparison table). ``_skipped`` and
    ``_notes`` are attached on ``df.attrs``.
    """
    from paired_slides_eval.evaluate import evaluate

    return _to_frame(evaluate(target, generated, **evaluate_kwargs), name)


def metrics_files(target, generated, *, name: str = "value", **evaluate_files_kwargs) -> "pd.DataFrame":
    """Same tidy table, but from file paths — wraps
    :func:`paired_slides_eval.evaluate.evaluate_files`.

    The path-based counterpart used to build cross-model tables from saved artifacts (a shared
    ``preprocess_pair`` ``.pkl`` target + each model's generated ``.h5ad``): space + coordinate
    reconciliation happen inside ``evaluate_files`` exactly as before.
    """
    from paired_slides_eval.evaluate import evaluate_files

    return _to_frame(evaluate_files(target, generated, **evaluate_files_kwargs), name)


def compare(named_pairs, *, from_files: bool = False, **kwargs) -> "pd.DataFrame":
    """Build a wide comparison table (metrics x models) from several models.

    ``named_pairs`` maps ``model_name -> (target, generated)``; each pair is scored and the columns
    concatenated. Set ``from_files=True`` to treat the pairs as file paths (uses ``metrics_files``).
    """
    import pandas as pd

    run = metrics_files if from_files else metrics
    cols = [run(t, g, name=str(model), **kwargs) for model, (t, g) in dict(named_pairs).items()]
    return pd.concat(cols, axis=1)
