"""Dataset registry — metadata and label mappings for all supported datasets.

This module defines the canonical information for every dataset the framework
supports.  The :pyclass:`DatasetInfo` dataclass captures schema details,
label names, and the optional unified-label mapping so that downstream code
never needs to hard-code dataset-specific knowledge.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class DatasetInfo:
    """Metadata descriptor for a single dataset."""

    name: str
    language: str
    source: str                          # "huggingface" | "local" | "github"
    hf_id: Optional[str] = None
    hf_config: Optional[str] = None
    path: Optional[str] = None           # For local datasets
    github_url: Optional[str] = None     # For GitHub-hosted datasets
    text_column: str = "text"
    label_column: str = "label"
    multilabel: bool = False
    # For multi-label binary-column format (e.g. EmoNoBa)
    binary_label_columns: List[str] = field(default_factory=list)
    label_names: List[str] = field(default_factory=list)
    num_labels: int = 0
    # Pre-split file paths (relative to data/raw/)
    split_files: Dict[str, str] = field(default_factory=dict)
    # Mapping from original label -> unified label (optional)
    unified_label_map: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Unified 6-label scheme
# ---------------------------------------------------------------------------
UNIFIED_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "love"]


# ─── GoEmotions (simplified) → unified mapping ──────────────────────────────
_GO_EMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

_GO_EMOTIONS_TO_UNIFIED = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "love",
    "confusion": "surprise",
    "curiosity": "surprise",
    "desire": "love",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "anger",
    "embarrassment": "fear",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "love",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "joy",  # Neutral → joy (or could be dropped)
}


# ─── dair-ai/emotion labels ────────────────────────────────────────────────
_EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

_EMOTION_TO_UNIFIED = {
    "sadness": "sadness",
    "joy": "joy",
    "love": "love",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
}


# ─── Hindi EmoInHindi ──────────────────────────────────────────────────────
_EMOIN_HINDI_LABELS = [
    "anticipation", "confident", "neutral", "hopeful", "anger", "sad",
    "joy", "compassion", "fear", "disgusted", "annoyed", "grateful",
    "impressed", "apprehensive", "surprised", "guilty", "confused",
]

_EMOIN_HINDI_TO_UNIFIED = {
    "joy": "joy",
    "hopeful": "joy",
    "confident": "joy",
    "grateful": "joy",
    "neutral": "joy",
    "sad": "sadness",
    "guilty": "sadness",
    "fear": "fear",
    "apprehensive": "fear",
    "anger": "anger",
    "annoyance": "anger",
    "annoyed": "anger",
    "disgusted": "anger",
    "surprised": "surprise",
    "surprise": "surprise",
    "anticipation": "surprise",
    "impressed": "surprise",
    "confused": "surprise",
    "compassion": "love",
}


# ─── Bengali EmoNoBa ──────────────────────────────────────────────────────
# EmoNoBa: 22,698 Bangla social media comments, 6 emotion categories.
# Multi-label format: binary columns (Love, Joy, Surprise, Anger, Sadness, Fear)
# Source: KhondokerIslam/EmoNoBa on GitHub
# Labels match our unified scheme exactly!
_EMONOBA_LABELS = ["love", "joy", "surprise", "anger", "sadness", "fear"]
_EMONOBA_BINARY_COLS = ["Love", "Joy", "Surprise", "Anger", "Sadness", "Fear"]

_EMONOBA_TO_UNIFIED = {
    "love": "love",
    "joy": "joy",
    "surprise": "surprise",
    "anger": "anger",
    "sadness": "sadness",
    "fear": "fear",
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, DatasetInfo] = {

    # ── English ───────────────────────────────────────────────────────────
    "go_emotions": DatasetInfo(
        name="go_emotions",
        language="english",
        source="huggingface",
        hf_id="google-research-datasets/go_emotions",
        hf_config="simplified",
        text_column="text",
        label_column="labels",
        multilabel=True,
        label_names=_GO_EMOTIONS_LABELS,
        num_labels=28,
        unified_label_map=_GO_EMOTIONS_TO_UNIFIED,
    ),

    "emotion": DatasetInfo(
        name="emotion",
        language="english",
        source="huggingface",
        hf_id="dair-ai/emotion",
        text_column="text",
        label_column="label",
        multilabel=False,
        label_names=_EMOTION_LABELS,
        num_labels=6,
        unified_label_map=_EMOTION_TO_UNIFIED,
    ),

    # ── Hindi ─────────────────────────────────────────────────────────────
    "emo_in_hindi": DatasetInfo(
        name="emo_in_hindi",
        language="hindi",
        source="local",
        path="data/raw/EmoInHindi/LREC_EmoInHindi.csv",
        text_column="utterance",
        label_column="emotions",
        multilabel=True,
        label_names=_EMOIN_HINDI_LABELS,
        num_labels=17,
        unified_label_map=_EMOIN_HINDI_TO_UNIFIED,
    ),

    "go_emotions_hindi": DatasetInfo(
        name="go_emotions_hindi",
        language="hindi",
        source="huggingface",
        hf_id="utkarsharora100/google_go_emotions_hindi_translated",
        text_column="text",
        label_column="label",
        multilabel=False,
        label_names=_GO_EMOTIONS_LABELS,
        num_labels=28,
        unified_label_map=_GO_EMOTIONS_TO_UNIFIED,
    ),

    # ── Bengali ───────────────────────────────────────────────────────────
    "emonoba": DatasetInfo(
        name="emonoba",
        language="bengali",
        source="github",
        github_url="https://github.com/KhondokerIslam/EmoNoBa",
        path="data/raw/emonoba",
        text_column="Data",             # Bengali text column name in CSV
        label_column=None,              # No single label column — uses binary cols
        multilabel=True,
        binary_label_columns=_EMONOBA_BINARY_COLS,
        label_names=_EMONOBA_LABELS,
        num_labels=6,
        split_files={
            "train": "Train.csv",
            "val": "Val.csv",
            "test": "Test.csv",
        },
        unified_label_map=_EMONOBA_TO_UNIFIED,
    ),

    # Keep old bengali_emotion for backward compat (generic local CSV)
    "bengali_emotion": DatasetInfo(
        name="bengali_emotion",
        language="bengali",
        source="local",
        path="data/raw/bengali_emotion.csv",
        text_column="text",
        label_column="label",
        multilabel=False,
        label_names=[],  # Populated at load time from the CSV
        num_labels=0,
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    """Look up a dataset by name.

    Raises
    ------
    KeyError
        If the dataset name is not in the registry.
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {available}"
        )
    return DATASET_REGISTRY[name]


def list_datasets(language: Optional[str] = None) -> List[DatasetInfo]:
    """List all registered datasets, optionally filtered by language."""
    if language is None:
        return list(DATASET_REGISTRY.values())
    return [d for d in DATASET_REGISTRY.values() if d.language == language]


def register_dataset(info: DatasetInfo) -> None:
    """Register a new dataset at runtime (e.g. for custom Bengali CSVs)."""
    DATASET_REGISTRY[info.name] = info
