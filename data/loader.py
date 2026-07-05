"""Unified dataset loader for multilingual emotion classification.

Handles loading from HuggingFace Hub, local CSV/JSON files, and GitHub-hosted
datasets (e.g. EmoNoBa). Applies label encoding, optional unified-label
mapping, and produces standardised ``EmotionDataset`` objects ready for
downstream consumption.
"""

import logging
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.registry import (
    DATASET_REGISTRY,
    UNIFIED_LABELS,
    DatasetInfo,
    get_dataset_info,
)
from data.splitter import stratified_split, print_split_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EmotionDataset — the canonical output of this module
# ---------------------------------------------------------------------------
@dataclass
class EmotionDataset:
    """Standardised container for an emotion classification dataset.

    Every loader path eventually produces one of these.
    """

    name: str
    language: str
    label_names: List[str]
    num_labels: int

    # Each split is {"texts": [...], "labels": [...]}
    train: Dict[str, list] = field(default_factory=dict)
    val: Dict[str, list] = field(default_factory=dict)
    test: Dict[str, list] = field(default_factory=dict)

    def summary(self) -> str:
        parts = [
            f"EmotionDataset(name={self.name}, lang={self.language}, "
            f"labels={self.num_labels})",
        ]
        for split_name in ("train", "val", "test"):
            data = getattr(self, split_name, {})
            n = len(data.get("texts", []))
            parts.append(f"  {split_name}: {n} samples")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Label encoding helpers
# ---------------------------------------------------------------------------
class LabelEncoder:
    """Bidirectional label ↔ integer mapping."""

    def __init__(self, label_names: Optional[List[str]] = None):
        self.label_names: List[str] = label_names or []
        self._label2id: Dict[str, int] = {}
        self._id2label: Dict[int, str] = {}
        if label_names:
            self._build(label_names)

    def _build(self, names: List[str]) -> None:
        self.label_names = list(names)
        self._label2id = {name: idx for idx, name in enumerate(names)}
        self._id2label = {idx: name for idx, name in enumerate(names)}

    def fit(self, labels: List[Any]) -> "LabelEncoder":
        """Learn label mapping from a list of raw labels."""
        unique = sorted(set(str(l) for l in labels))
        self._build(unique)
        return self

    def encode(self, label: Any) -> int:
        """Encode a single label to an integer."""
        key = str(label)
        if key not in self._label2id:
            # Auto-extend (for unseen labels at inference time)
            idx = len(self.label_names)
            self.label_names.append(key)
            self._label2id[key] = idx
            self._id2label[idx] = key
            logger.warning(f"Unseen label '{key}' auto-assigned to {idx}")
        return self._label2id[key]

    def encode_batch(self, labels: List[Any]) -> List[int]:
        return [self.encode(l) for l in labels]

    def decode(self, idx: int) -> str:
        return self._id2label.get(idx, f"UNKNOWN_{idx}")

    def decode_batch(self, indices: List[int]) -> List[str]:
        return [self.decode(i) for i in indices]

    @property
    def num_labels(self) -> int:
        return len(self.label_names)


# ---------------------------------------------------------------------------
# Internal loading functions
# ---------------------------------------------------------------------------
def _load_huggingface(info: DatasetInfo) -> Dict[str, pd.DataFrame]:
    """Load a dataset from HuggingFace Hub, return per-split DataFrames."""
    from datasets import load_dataset

    logger.info(f"Loading HuggingFace dataset: {info.hf_id} (config={info.hf_config})")

    kwargs = {"path": info.hf_id}
    if info.hf_config:
        kwargs["name"] = info.hf_config

    try:
        ds = load_dataset(**kwargs, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset {info.hf_id}: {e}")
        raise

    result = {}
    for split_name in ("train", "validation", "test"):
        if split_name in ds:
            df = ds[split_name].to_pandas()
            result[split_name] = df
            logger.info(f"  {split_name}: {len(df)} rows, "
                         f"columns={list(df.columns)}")

    # If there's no explicit validation split, we'll handle it later
    if "validation" not in result and "val" not in result:
        logger.info("  No validation split found — will create from train.")

    return result


def _load_local(info: DatasetInfo) -> Dict[str, pd.DataFrame]:
    """Load a local CSV or JSON file."""
    path = info.path
    if not os.path.isabs(path):
        # Make relative to project root
        from config import get_project_root
        path = str(get_project_root() / path)

    logger.info(f"Loading local dataset: {path}")

    if not os.path.exists(path):
        logger.error(f"Dataset file not found: {path}")
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.endswith(".csv"):
        df = pd.read_csv(path, encoding="utf-8")
    elif path.endswith(".json") or path.endswith(".jsonl"):
        df = pd.read_json(path, lines=path.endswith(".jsonl"), encoding="utf-8")
    elif path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {path}")

    logger.info(f"  Loaded {len(df)} rows, columns={list(df.columns)}")

    # Return as single "train" split — will be split later
    return {"train": df}


def _download_emonoba(info: DatasetInfo) -> str:
    """Download EmoNoBa dataset from GitHub if not already cached.

    Returns the path to the directory containing Train.csv, Val.csv, Test.csv.
    """
    data_dir = info.path
    if not os.path.isabs(data_dir):
        from config import get_project_root
        data_dir = str(get_project_root() / data_dir)

    # Check if already downloaded
    if os.path.isdir(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        if csv_files:
            logger.info(f"EmoNoBa already cached at: {data_dir} ({len(csv_files)} CSVs)")
            return data_dir

    os.makedirs(data_dir, exist_ok=True)

    # Download from GitHub as zip
    github_url = "https://github.com/KhondokerIslam/EmoNoBa/archive/refs/heads/main.zip"
    zip_path = os.path.join(data_dir, "emonoba.zip")

    logger.info(f"Downloading EmoNoBa from GitHub...")
    try:
        import urllib.request
        urllib.request.urlretrieve(github_url, zip_path)
    except Exception as e:
        logger.error(
            f"Failed to download EmoNoBa: {e}\n"
            f"Please manually download from {info.github_url} and place "
            f"Train.csv, Val.csv, Test.csv in {data_dir}"
        )
        raise

    # Extract CSVs from the zip
    logger.info(f"Extracting EmoNoBa...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Look for CSV files (they might be in a subdirectory)
            if member.endswith(".csv"):
                basename = os.path.basename(member)
                target = os.path.join(data_dir, basename)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                logger.info(f"  Extracted: {basename}")

    # Clean up zip
    os.remove(zip_path)
    logger.info(f"EmoNoBa downloaded to: {data_dir}")
    return data_dir


def _load_split_csvs(info: DatasetInfo) -> Dict[str, pd.DataFrame]:
    """Load pre-split CSV files (e.g. EmoNoBa Train.csv, Val.csv, Test.csv)."""
    data_dir = info.path
    if not os.path.isabs(data_dir):
        from config import get_project_root
        data_dir = str(get_project_root() / data_dir)

    result = {}
    split_mapping = {
        "train": info.split_files.get("train", "Train.csv"),
        "validation": info.split_files.get("val", "Val.csv"),
        "test": info.split_files.get("test", "Test.csv"),
    }

    for split_name, filename in split_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, encoding="utf-8")
            result[split_name] = df
            logger.info(f"  {split_name}: {len(df)} rows, columns={list(df.columns)}")
        else:
            logger.warning(f"  Split file not found: {filepath}")

    return result


def _extract_binary_column_labels(
    df: pd.DataFrame,
    text_column: str,
    binary_label_columns: List[str],
    label_names: List[str],
) -> Tuple[List[str], List[Any]]:
    """Extract texts and labels from binary-column multi-label format.

    EmoNoBa format: columns like Love(0/1), Joy(0/1), etc.
    For single-label classification, takes the FIRST active emotion.
    If no emotion is active, assigns 'joy' as fallback.

    Parameters
    ----------
    df : pd.DataFrame
    text_column : str
        Column containing the text (e.g. 'Data' for EmoNoBa).
    binary_label_columns : list[str]
        Column names for binary emotion flags (e.g. ['Love', 'Joy', ...]).
    label_names : list[str]
        Corresponding lowercase label names (e.g. ['love', 'joy', ...]).

    Returns
    -------
    tuple[list[str], list[str]]
        (texts, labels_as_strings)
    """
    texts = df[text_column].astype(str).tolist()
    labels = []

    for _, row in df.iterrows():
        # Find which emotions are active (value == 1)
        active = []
        for col, lbl_name in zip(binary_label_columns, label_names):
            if col in df.columns and row.get(col, 0) == 1:
                active.append(lbl_name)

        if active:
            # Take the first active emotion as the primary label
            labels.append(active[0])
        else:
            # No emotion active — default to the first label
            labels.append(label_names[0] if label_names else "unknown")

    return texts, labels


def _extract_texts_labels(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    multilabel: bool = False,
) -> Tuple[List[str], List[Any]]:
    """Extract text and label columns from a DataFrame.

    For multi-label datasets, takes the first label for simplicity.
    """
    texts = df[text_column].astype(str).tolist()

    if multilabel:
        # GoEmotions: labels column is a list of ints
        raw_labels = df[label_column].tolist()
        # Take the first label (primary emotion)
        labels = []
        for lbl in raw_labels:
            if isinstance(lbl, list) and len(lbl) > 0:
                labels.append(lbl[0])
            elif isinstance(lbl, (int, np.integer)):
                labels.append(int(lbl))
            elif isinstance(lbl, str):
                parts = [p.strip() for p in lbl.split(",") if p.strip()]
                labels.append(parts[0] if parts else "neutral")
            else:
                labels.append(0)
    else:
        labels = df[label_column].tolist()

    return texts, labels


def _apply_unified_mapping(
    labels: List[Any],
    label_names: List[str],
    unified_map: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    """Map original labels to unified label set.

    Parameters
    ----------
    labels : list
        Original labels (could be int indices or strings).
    label_names : list[str]
        Original label name list for decoding int labels.
    unified_map : dict
        Mapping from original label name → unified label name.

    Returns
    -------
    tuple[list[str], list[str]]
        (mapped_labels_as_strings, unified_label_names)
    """
    mapped = []
    for lbl in labels:
        if isinstance(lbl, (int, np.integer)) and label_names:
            if int(lbl) < len(label_names):
                original_name = label_names[int(lbl)]
            else:
                original_name = str(lbl)
        else:
            original_name = str(lbl)

        unified_name = unified_map.get(original_name, original_name)
        mapped.append(unified_name)

    return mapped, UNIFIED_LABELS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_dataset_by_name(
    dataset_name: str,
    label_mode: str = "per_dataset",
    split_ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> EmotionDataset:
    """Load a dataset by its registry name.

    Parameters
    ----------
    dataset_name : str
        Key in ``DATASET_REGISTRY`` (e.g. ``"emotion"``, ``"emo_in_hindi"``).
    label_mode : str
        ``"per_dataset"`` keeps original labels; ``"unified"`` maps to the
        6-label common set.
    split_ratios : dict, optional
        ``{"train": 0.8, "val": 0.1, "test": 0.1}``.
    seed : int
        Random seed for splitting.
    max_samples : int, optional
        Limit total samples (for debugging).

    Returns
    -------
    EmotionDataset
    """
    info = get_dataset_info(dataset_name)
    split_ratios = split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1}

    # 1. Load raw data
    if info.source == "huggingface":
        raw_splits = _load_huggingface(info)
    elif info.source == "github":
        # Download if needed, then load split CSVs
        _download_emonoba(info)
        raw_splits = _load_split_csvs(info)
    elif info.source == "local":
        raw_splits = _load_local(info)
    else:
        raise ValueError(f"Unknown source: {info.source}")

    # 2. Extract texts and labels per split
    split_data: Dict[str, Dict[str, list]] = {}

    for split_name, df in raw_splits.items():
        if max_samples:
            df = df.head(max_samples)

        # Use binary column extraction for EmoNoBa-style datasets
        if info.binary_label_columns:
            texts, labels = _extract_binary_column_labels(
                df, info.text_column,
                info.binary_label_columns, info.label_names,
            )
        else:
            texts, labels = _extract_texts_labels(
                df, info.text_column, info.label_column, info.multilabel
            )
        split_data[split_name] = {"texts": texts, "labels": labels}

    # 3. Handle missing splits
    canonical = {}  # train, val, test

    if "train" in split_data and "test" in split_data:
        canonical["train"] = split_data["train"]
        canonical["test"] = split_data["test"]

        if "validation" in split_data:
            canonical["val"] = split_data["validation"]
        elif "val" in split_data:
            canonical["val"] = split_data["val"]
        else:
            # Split a portion of train for validation
            all_texts = canonical["train"]["texts"]
            all_labels = canonical["train"]["labels"]
            val_size = split_ratios.get("val", 0.1)

            from sklearn.model_selection import train_test_split
            try:
                t_texts, v_texts, t_labels, v_labels = train_test_split(
                    all_texts, all_labels,
                    test_size=val_size,
                    random_state=seed,
                    stratify=all_labels,
                )
            except ValueError:
                logger.warning("Stratified split for validation failed (likely due to sparse classes) — falling back to random split.")
                t_texts, v_texts, t_labels, v_labels = train_test_split(
                    all_texts, all_labels,
                    test_size=val_size,
                    random_state=seed,
                    stratify=None,
                )
            canonical["train"] = {"texts": t_texts, "labels": t_labels}
            canonical["val"] = {"texts": v_texts, "labels": v_labels}
    else:
        # Only have one split — do full stratified split
        all_texts = split_data.get("train", split_data[list(split_data.keys())[0]])["texts"]
        all_labels = split_data.get("train", split_data[list(split_data.keys())[0]])["labels"]

        if max_samples:
            all_texts = all_texts[:max_samples]
            all_labels = all_labels[:max_samples]

        canonical = stratified_split(
            all_texts, all_labels,
            train_ratio=split_ratios["train"],
            val_ratio=split_ratios["val"],
            test_ratio=split_ratios["test"],
            seed=seed,
        )

    # 4. Determine label names and apply unified mapping if needed
    label_names = list(info.label_names) if info.label_names else []

    if label_mode == "unified" and info.unified_label_map:
        # Apply unified mapping to all splits
        for split_name in canonical:
            raw_labels = canonical[split_name]["labels"]
            mapped, _ = _apply_unified_mapping(
                raw_labels, label_names, info.unified_label_map
            )
            canonical[split_name]["labels"] = mapped
        label_names = list(UNIFIED_LABELS)

    # 5. Encode labels to integers
    encoder = LabelEncoder()

    if label_names:
        encoder._build(label_names)
    else:
        # Learn from all labels
        all_labels_raw = []
        for split_data_inner in canonical.values():
            all_labels_raw.extend(split_data_inner["labels"])
        encoder.fit(all_labels_raw)
        label_names = encoder.label_names

    for split_name in canonical:
        canonical[split_name]["labels"] = encoder.encode_batch(
            canonical[split_name]["labels"]
        )

    # 6. Build EmotionDataset
    dataset = EmotionDataset(
        name=dataset_name,
        language=info.language,
        label_names=label_names,
        num_labels=encoder.num_labels,
        train=canonical.get("train", {}),
        val=canonical.get("val", {}),
        test=canonical.get("test", {}),
    )

    logger.info(f"\n{dataset.summary()}")
    print_split_stats(canonical, label_names)

    return dataset


def load_dataset_for_language(
    language: str,
    config,
    max_samples: Optional[int] = None,
) -> EmotionDataset:
    """Load the active dataset for a given language using framework config.

    Parameters
    ----------
    language : str
        ``"english"``, ``"hindi"``, or ``"bengali"``.
    config : Config
        Framework configuration object.
    max_samples : int, optional
        Limit dataset size for quick testing.

    Returns
    -------
    EmotionDataset
    """
    # Get dataset list for this language
    datasets_for_lang = config.datasets.get(language, [])
    if not datasets_for_lang:
        raise ValueError(f"No datasets configured for language: {language}")

    # Get active index
    active_idx = config.active_dataset.get(language, 0)
    if isinstance(active_idx, int) and active_idx < len(datasets_for_lang):
        ds_cfg = datasets_for_lang[active_idx]
    else:
        ds_cfg = datasets_for_lang[0]

    dataset_name = ds_cfg.name if hasattr(ds_cfg, "name") else ds_cfg.get("name", "")

    # Get preprocessing config
    label_mode = "per_dataset"
    split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    seed = 42

    if hasattr(config, "preprocessing"):
        pp = config.preprocessing
        label_mode = pp.get("label_mode", "per_dataset")
        if hasattr(pp, "split_ratios"):
            sr = pp.split_ratios
            split_ratios = {
                "train": sr.get("train", 0.8),
                "val": sr.get("val", 0.1),
                "test": sr.get("test", 0.1),
            }

    if hasattr(config, "seed"):
        seed = config.seed

    return load_dataset_by_name(
        dataset_name=dataset_name,
        label_mode=label_mode,
        split_ratios=split_ratios,
        seed=seed,
        max_samples=max_samples,
    )


def load_all_datasets(
    config,
    languages: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, EmotionDataset]:
    """Load datasets for multiple languages.

    Parameters
    ----------
    config : Config
        Framework configuration.
    languages : list[str], optional
        Languages to load. Defaults to all configured languages.
    max_samples : int, optional
        Limit per dataset.

    Returns
    -------
    dict[str, EmotionDataset]
        Mapping from language → EmotionDataset.
    """
    if languages is None:
        languages = list(config.datasets.to_dict().keys()) if hasattr(config.datasets, "to_dict") else ["english"]

    results = {}
    for lang in languages:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Loading dataset for: {lang.upper()}")
            logger.info(f"{'='*60}")
            results[lang] = load_dataset_for_language(lang, config, max_samples)
        except Exception as e:
            logger.error(f"Failed to load dataset for {lang}: {e}")
            logger.warning(f"Skipping {lang}...")

    return results
