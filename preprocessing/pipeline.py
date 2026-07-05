"""Preprocessing pipeline orchestrator for multilingual emotion datasets.

Chains text cleaning, Unicode normalization, and optional tokenization
into a single ``preprocess_dataset`` call that accepts an
:class:`~data.loader.EmotionDataset` and returns a new one with cleaned
text.
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

from data.loader import EmotionDataset
from preprocessing.cleaner import clean_text
from preprocessing.unicode_normalizer import normalize_text
from preprocessing.tokenizer import tokenize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level: clean a list of texts
# ---------------------------------------------------------------------------
def preprocess_texts(
    texts: List[str],
    language: str = "english",
    config: Optional[object] = None,
) -> List[str]:
    """Clean and normalize a list of raw texts.

    Applies, in order:
    1. :func:`clean_text` (URL/HTML/emoji/special-char removal, lowercasing)
    2. :func:`normalize_text` (Unicode normalization + script-specific fixes)

    Parameters
    ----------
    texts : list[str]
        Raw text strings.
    language : str
        ``"english"``, ``"hindi"``, or ``"bengali"``.
    config : Config, optional
        Framework configuration object.

    Returns
    -------
    list[str]
        Cleaned, normalized texts.
    """
    unicode_form = _get_unicode_form(config)

    cleaned: List[str] = []
    for text in texts:
        t = clean_text(text, language=language, config=config)
        t = normalize_text(t, language=language, form=unicode_form)
        cleaned.append(t)

    return cleaned


# ---------------------------------------------------------------------------
# High-level: preprocess an EmotionDataset
# ---------------------------------------------------------------------------
def preprocess_dataset(
    dataset: EmotionDataset,
    config: Optional[object] = None,
) -> EmotionDataset:
    """Preprocess all splits of an :class:`EmotionDataset`.

    Creates a **new** ``EmotionDataset`` with cleaned text.  Optionally
    stores tokenized versions alongside the cleaned text in each split
    dict (key ``"tokens"``).

    Parameters
    ----------
    dataset : EmotionDataset
        Dataset loaded by :func:`data.loader.load_dataset_by_name`.
    config : Config, optional
        Framework configuration object.

    Returns
    -------
    EmotionDataset
        A new dataset with preprocessed text.
    """
    language = dataset.language
    logger.info(
        "Preprocessing dataset '%s' (language=%s) …", dataset.name, language
    )

    tok_method = _get_tokenizer_method(config, language)
    do_tokenize = tok_method is not None

    new_splits: Dict[str, Dict[str, list]] = {}

    for split_name in ("train", "val", "test"):
        split_data: Dict[str, list] = getattr(dataset, split_name, {})
        if not split_data or "texts" not in split_data:
            new_splits[split_name] = split_data
            continue

        raw_texts = split_data["texts"]
        cleaned = preprocess_texts(raw_texts, language=language, config=config)

        new_split: Dict[str, list] = {
            "texts": cleaned,
            "labels": list(split_data.get("labels", [])),
        }

        if do_tokenize:
            tokens = [
                tokenize(t, method=tok_method, language=language)
                for t in cleaned
            ]
            new_split["tokens"] = tokens

        new_splits[split_name] = new_split

        n = len(cleaned)
        logger.info("  %s: %d samples cleaned.", split_name, n)

    result = EmotionDataset(
        name=dataset.name,
        language=dataset.language,
        label_names=list(dataset.label_names),
        num_labels=dataset.num_labels,
        train=new_splits.get("train", {}),
        val=new_splits.get("val", {}),
        test=new_splits.get("test", {}),
    )

    logger.info("Preprocessing complete for '%s'.", dataset.name)
    return result


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _get_unicode_form(config: Optional[object]) -> str:
    """Read ``config.preprocessing.unicode_form`` with fallback."""
    if config is None:
        return "NFC"
    pp = getattr(config, "preprocessing", None)
    if pp is None:
        return "NFC"
    form = getattr(pp, "unicode_form", None)
    if form is None and hasattr(pp, "get"):
        form = pp.get("unicode_form", None)
    return str(form) if form else "NFC"


def _get_tokenizer_method(
    config: Optional[object], language: str
) -> Optional[str]:
    """Read the tokenizer method for *language* from config.

    Returns ``None`` when no tokenizer config is found (tokenization
    will be skipped).
    """
    if config is None:
        return None
    pp = getattr(config, "preprocessing", None)
    if pp is None:
        return None

    tok_cfg = getattr(pp, "tokenizer", None)
    if tok_cfg is None:
        return None

    # tok_cfg may be a Config object or a dict-like
    method = getattr(tok_cfg, language.lower(), None)
    if method is None and hasattr(tok_cfg, "get"):
        method = tok_cfg.get(language.lower(), None)

    return str(method) if method else None
