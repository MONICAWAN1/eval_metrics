"""Cell-type classifier: nets + full training pipeline.

Imports are **lazy** (PEP 562): the nets (`nets.py`) need only torch / torch_geometric, while the
task / dataset / datamodule pull in lightning. Resolving symbols on demand means the concordance
metric can `from nicheflow_eval.classifier.nets import SpatialCTClassifierNet` without dragging in
lightning, and importing the package never triggers the training stack until you actually use it.
"""

import importlib

_EXPORTS = {
    # torch-only (used by the concordance metric)
    "CTClassifierNet": "nets",
    "SpatialCTClassifierBase": "nets",
    "SpatialCTClassifierNet": "nets",
    "SpatialDeepSetCTClassifierNet": "nets",
    "GraphCTClassifierNet": "nets",
    # lightning-dependent (training only)
    "CellTypeClassification": "task",
    "Plots": "task",
    "H5ADCTDataset": "dataset",
    "SpatialH5ADCTDataset": "dataset",
    "CellTypeBatch": "dataset",
    "H5ADCTDataModule": "datamodule",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        module = importlib.import_module(f"{__name__}.{_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
