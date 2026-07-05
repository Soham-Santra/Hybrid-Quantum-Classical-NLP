"""Tokenization utilities for multilingual text processing.

Supports multiple tokenization methods:
- ``whitespace``: simple ``str.split()``
- ``nltk``: NLTK ``word_tokenize`` (English only, with auto-download)
- ``basic``: regex-based tokenizer suitable for any language

Does **not** depend on ``indic_nlp_library`` (which may not be available
in Google Colab) — uses the ``basic`` regex tokenizer as a safe fallback
for Hindi and Bengali.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex pattern for the 'basic' tokenizer
# ---------------------------------------------------------------------------
# Matches sequences of:
#   - word characters in any script (\w+)
#   - or standalone non-whitespace, non-word chars (punctuation, etc.)
_RE_TOKEN = re.compile(r"\w+|[^\s\w]", re.UNICODE)

# Supported methods
_METHODS = {"whitespace", "nltk", "basic"}

# Languages for which NLTK word_tokenize is appropriate
_NLTK_LANGUAGES = {"english", "en"}


def tokenize(
    text: str,
    method: str = "whitespace",
    language: str = "english",
) -> List[str]:
    """Tokenize *text* into a list of string tokens.

    Parameters
    ----------
    text : str
        Input text (should already be cleaned).
    method : str
        Tokenization strategy: ``"whitespace"``, ``"nltk"``, or ``"basic"``.
    language : str
        Language hint (``"english"``, ``"hindi"``, ``"bengali"``).
        Used to choose NLTK locale and to auto-fallback when NLTK is
        unavailable for non-English text.

    Returns
    -------
    list[str]
        List of tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    method = method.lower()
    if method not in _METHODS:
        logger.warning(
            "Unknown tokenization method '%s', falling back to 'basic'.",
            method,
        )
        method = "basic"

    # NLTK requested for a non-English language → fall back to basic
    if method == "nltk" and language.lower() not in _NLTK_LANGUAGES:
        logger.debug(
            "NLTK tokenizer not suitable for '%s'; using 'basic' instead.",
            language,
        )
        method = "basic"

    if method == "whitespace":
        return _tokenize_whitespace(text)
    elif method == "nltk":
        return _tokenize_nltk(text)
    else:  # basic
        return _tokenize_basic(text)


# ---------------------------------------------------------------------------
# Individual tokenizers
# ---------------------------------------------------------------------------
def _tokenize_whitespace(text: str) -> List[str]:
    """Split on whitespace."""
    return text.split()


def _tokenize_basic(text: str) -> List[str]:
    """Regex-based tokenizer that works across scripts."""
    return _RE_TOKEN.findall(text)


def _tokenize_nltk(text: str) -> List[str]:
    """Tokenize using NLTK ``word_tokenize``.

    Automatically downloads the ``punkt_tab`` data on first use if
    missing.  Falls back to the basic regex tokenizer if NLTK is not
    installed.
    """
    try:
        import nltk  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "NLTK not installed; falling back to basic regex tokenizer."
        )
        return _tokenize_basic(text)

    # Ensure punkt_tab resource is available
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK 'punkt_tab' tokenizer data …")
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to download NLTK punkt_tab: %s. "
                "Falling back to basic tokenizer.",
                exc,
            )
            return _tokenize_basic(text)

    try:
        return nltk.word_tokenize(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "NLTK word_tokenize failed: %s. Falling back to basic.", exc
        )
        return _tokenize_basic(text)
