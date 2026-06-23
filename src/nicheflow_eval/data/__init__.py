"""Data contract and dataset-dataclass loader for the evaluation metrics."""

from nicheflow_eval.data.dataclass import (
    H5ADDatasetDataclass,
    load_h5ad_dataset_dataclass,
)

__all__ = ["H5ADDatasetDataclass", "load_h5ad_dataset_dataclass"]
