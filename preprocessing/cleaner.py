"""Text cleaning utilities for multilingual emotion classification.

Provides language-aware cleaning for English, Hindi, and Bengali text
including URL removal, HTML stripping, special character handling,
emoji processing, and script-aware lowercasing.
"""

import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for reuse)
# ---------------------------------------------------------------------------
_RE_URL = re.compile(r"http\S+", re.UNICODE)
_RE_HTML = re.compile(r"<.*?>", re.DOTALL)
_RE_SPECIAL = re.compile(r"[^\w\s]", re.UNICODE)
_RE_WHITESPACE = re.compile(r"\s+")

# Emoji pattern — covers the main Unicode emoji blocks
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc symbols & pictographs
    "\U0001F680-\U0001F6FF"  # Transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "\U0001F900-\U0001F9FF"  # Supplemental symbols
    "\U0001FA00-\U0001FA6F"  # Chess symbols
    "\U0001FA70-\U0001FAFF"  # Symbols extended-A
    "]+",
    flags=re.UNICODE,
)

# Indic script ranges for detecting Hindi/Bengali characters
_DEVANAGARI_RANGE = range(0x0900, 0x097F + 1)
_BENGALI_RANGE = range(0x0980, 0x09FF + 1)

_INDIC_LANGUAGES = {"hindi", "bengali", "bn", "hi"}


def _should_lowercase(language: str) -> bool:
    """Return True if lowercasing is appropriate for *language*."""
    return language.lower() not in _INDIC_LANGUAGES


def clean_text(
    text: str,
    language: str = "english",
    config: Optional[object] = None,
) -> str:
    """Clean a single text string with language-aware rules.

    Parameters
    ----------
    text : str
        Raw input text.
    language : str
        ``"english"``, ``"hindi"``, or ``"bengali"``.
    config : Config, optional
        Framework configuration object (dot-access).  When ``None``,
        sensible defaults are used.

    Returns
    -------
    str
        Cleaned text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # ── Extract config flags with safe defaults ──────────────────────────
    remove_urls: bool = _cfg_flag(config, "remove_urls", True)
    remove_html: bool = _cfg_flag(config, "remove_html", True)
    remove_special: bool = _cfg_flag(config, "remove_special_chars", False)
    remove_emojis: bool = _cfg_flag(config, "remove_emojis", False)
    do_lowercase: bool = _cfg_flag(config, "lowercase", True)

    # ── Pipeline steps ───────────────────────────────────────────────────
    if remove_urls:
        text = _RE_URL.sub("", text)

    if remove_html:
        text = _RE_HTML.sub("", text)

    if remove_emojis:
        text = _RE_EMOJI.sub("", text)

    if remove_special:
        text = _RE_SPECIAL.sub(" ", text)

    # Lowercase only for scripts that have case distinction
    if do_lowercase and _should_lowercase(language):
        text = text.lower()

    # Collapse multiple whitespace and strip
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cfg_flag(config: Optional[object], key: str, default: bool) -> bool:
    """Safely read a boolean flag from *config.preprocessing.<key>*.

    Supports both direct attribute access on config (when config is
    already the preprocessing sub-object) and nested access via
    ``config.preprocessing``.
    """
    if config is None:
        return default

    # Try config.preprocessing.<key> first
    pp = getattr(config, "preprocessing", None)
    if pp is not None:
        val = getattr(pp, key, None)
        if val is None and hasattr(pp, "get"):
            val = pp.get(key, None)
        if val is not None:
            return bool(val)

    # Fall back to config.<key> directly
    val = getattr(config, key, None)
    if val is None and hasattr(config, "get"):
        val = config.get(key, None)
    if val is not None:
        return bool(val)

    return default
