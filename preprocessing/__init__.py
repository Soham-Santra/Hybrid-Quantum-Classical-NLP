"""Text preprocessing pipeline for multilingual NLP.

Public API
----------
- :func:`preprocess_dataset` — preprocess all splits of an EmotionDataset
- :func:`preprocess_texts` — clean and normalize a list of raw texts
- :func:`clean_text` — language-aware text cleaning
- :func:`normalize_unicode` — standard Unicode normalization
- :func:`tokenize` — multi-method tokenization
"""

from preprocessing.cleaner import clean_text
from preprocessing.pipeline import preprocess_dataset, preprocess_texts
from preprocessing.tokenizer import tokenize
from preprocessing.unicode_normalizer import normalize_unicode

__all__ = [
    "preprocess_dataset",
    "preprocess_texts",
    "clean_text",
    "normalize_unicode",
    "tokenize",
]
