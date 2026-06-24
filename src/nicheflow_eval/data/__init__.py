"""Data representations.

Two layers live here: :mod:`nicheflow_eval.data.anndata` is the model-agnostic AnnData -> arrays
IO that every metric uses (raw genes + ``obsm['spatial']``, optional PCA); ``dataclass`` below is
the internal **niche pickle schema** shared by the classifier-training dataset and the NicheFlow
adapter (it is not part of the user-facing AnnData input contract).
"""

from nicheflow_eval.data.dataclass import (
    H5ADDatasetDataclass,
    load_h5ad_dataset_dataclass,
)

__all__ = ["H5ADDatasetDataclass", "load_h5ad_dataset_dataclass"]
