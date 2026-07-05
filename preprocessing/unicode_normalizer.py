"""Unicode normalization for multilingual text processing.

Provides general Unicode normalization (NFC/NFKC/NFD/NFKD) plus
language-specific handling for Bengali and Hindi scripts to
canonicalise variant character representations.
"""

import logging
import re
import unicodedata
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid normalization forms
# ---------------------------------------------------------------------------
_VALID_FORMS = {"NFC", "NFKC", "NFD", "NFKD"}


# ---------------------------------------------------------------------------
# General Unicode normalization
# ---------------------------------------------------------------------------
def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Apply standard Unicode normalization.

    Parameters
    ----------
    text : str
        Input text.
    form : str
        Normalization form: ``"NFC"``, ``"NFKC"``, ``"NFD"``, or ``"NFKD"``.

    Returns
    -------
    str
        Normalized text.
    """
    if not text:
        return text

    form = form.upper()
    if form not in _VALID_FORMS:
        logger.warning(
            "Invalid normalization form '%s', falling back to NFC.", form
        )
        form = "NFC"

    return unicodedata.normalize(form, text)


# ---------------------------------------------------------------------------
# Bengali-specific normalization
# ---------------------------------------------------------------------------
# Common Bengali nukta / variant mappings.
# E.g. ড (U+09A1) + nukta (U+09BC) → ড় (U+09DC)
_BENGALI_NUKTA_MAP: Dict[str, str] = {
    "\u09A1\u09BC": "\u09DC",  # ড + nukta → ড়
    "\u09A2\u09BC": "\u09DD",  # ঢ + nukta → ঢ়
    "\u09AF\u09BC": "\u09DF",  # য + nukta → য়
}

# Pattern to find Bengali base + nukta sequences
_RE_BENGALI_NUKTA = re.compile(
    "|".join(re.escape(k) for k in _BENGALI_NUKTA_MAP)
)

# Canonicalise Bengali digit variants to standard Bengali digits
_BENGALI_DIGIT_MAP: Dict[str, str] = {}  # Extend if needed


def normalize_bengali(text: str) -> str:
    """Apply Bengali-specific normalization.

    Handles nukta variant merging (e.g. ড + ্ → ড়) and other
    common canonicalisation rules for Bengali text.

    Parameters
    ----------
    text : str
        Bengali text (may be mixed-script).

    Returns
    -------
    str
        Normalized text.
    """
    if not text:
        return text

    # Step 1: standard NFC first (compose precomposed forms)
    text = unicodedata.normalize("NFC", text)

    # Step 2: apply nukta contractions
    if _RE_BENGALI_NUKTA.pattern:
        text = _RE_BENGALI_NUKTA.sub(
            lambda m: _BENGALI_NUKTA_MAP[m.group(0)], text
        )

    # Step 3: remove zero-width joiners/non-joiners that sometimes appear
    text = text.replace("\u200D", "")  # ZWJ
    text = text.replace("\u200C", "")  # ZWNJ

    return text


# ---------------------------------------------------------------------------
# Hindi-specific normalization
# ---------------------------------------------------------------------------
# Common Devanagari nukta variants:
# क (U+0915) + nukta (U+093C) → क़ (U+0958), etc.
_HINDI_NUKTA_MAP: Dict[str, str] = {
    "\u0915\u093C": "\u0958",  # क + nukta → क़
    "\u0916\u093C": "\u0959",  # ख + nukta → ख़
    "\u0917\u093C": "\u095A",  # ग + nukta → ग़
    "\u091C\u093C": "\u095B",  # ज + nukta → ज़
    "\u0921\u093C": "\u095C",  # ड + nukta → ड़
    "\u0922\u093C": "\u095D",  # ढ + nukta → ढ़
    "\u092B\u093C": "\u095E",  # फ + nukta → फ़
    "\u092F\u093C": "\u095F",  # य + nukta → य़
}

_RE_HINDI_NUKTA = re.compile(
    "|".join(re.escape(k) for k in _HINDI_NUKTA_MAP)
)


def normalize_hindi(text: str) -> str:
    """Apply Hindi / Devanagari-specific normalization.

    Handles nukta variant merging for Devanagari consonants and
    removes stray joiners.

    Parameters
    ----------
    text : str
        Hindi text (may be mixed-script).

    Returns
    -------
    str
        Normalized text.
    """
    if not text:
        return text

    # Step 1: standard NFC
    text = unicodedata.normalize("NFC", text)

    # Step 2: merge nukta combinations
    if _RE_HINDI_NUKTA.pattern:
        text = _RE_HINDI_NUKTA.sub(
            lambda m: _HINDI_NUKTA_MAP[m.group(0)], text
        )

    # Step 3: remove stray joiners
    text = text.replace("\u200D", "")  # ZWJ
    text = text.replace("\u200C", "")  # ZWNJ

    return text


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def normalize_text(
    text: str,
    language: str = "english",
    form: str = "NFC",
) -> str:
    """Normalize text with language-aware rules.

    For Hindi and Bengali, applies script-specific normalization
    *after* general Unicode normalization.  For other languages,
    only standard Unicode normalization is applied.

    Parameters
    ----------
    text : str
        Input text.
    language : str
        ``"english"``, ``"hindi"``, or ``"bengali"``.
    form : str
        Unicode normalization form (default ``"NFC"``).

    Returns
    -------
    str
        Normalized text.
    """
    if not text:
        return text

    # General normalization first
    text = normalize_unicode(text, form=form)

    lang_lower = language.lower()
    if lang_lower in ("hindi", "hi"):
        text = normalize_hindi(text)
    elif lang_lower in ("bengali", "bn", "bangla"):
        text = normalize_bengali(text)

    return text
