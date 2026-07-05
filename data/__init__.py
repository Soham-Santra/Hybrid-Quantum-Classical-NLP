"""Data loading and dataset management.

This package provides unified dataset loading for multilingual emotion
classification across English, Hindi, and Bengali.

Quick Start
-----------
>>> from data.loader import load_dataset_by_name, load_dataset_for_language
>>> dataset = load_dataset_by_name("emotion")
>>> print(dataset.summary())

>>> from config import load_config
>>> cfg = load_config()
>>> dataset = load_dataset_for_language("english", cfg)
"""

from data.loader import (
    EmotionDataset,
    LabelEncoder,
    load_all_datasets,
    load_dataset_by_name,
    load_dataset_for_language,
)
from data.registry import (
    DATASET_REGISTRY,
    UNIFIED_LABELS,
    DatasetInfo,
    get_dataset_info,
    list_datasets,
    register_dataset,
)
from data.splitter import (
    get_label_distribution,
    stratified_split,
    subsample,
)

__all__ = [
    # Loader
    "EmotionDataset",
    "LabelEncoder",
    "load_all_datasets",
    "load_dataset_by_name",
    "load_dataset_for_language",
    # Registry
    "DATASET_REGISTRY",
    "UNIFIED_LABELS",
    "DatasetInfo",
    "get_dataset_info",
    "list_datasets",
    "register_dataset",
    # Splitter
    "get_label_distribution",
    "stratified_split",
    "subsample",
]
