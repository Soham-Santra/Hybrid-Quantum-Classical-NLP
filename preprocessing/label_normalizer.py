"""Label normalization for multilingual emotion classification.

Supports two modes:
- ``per_dataset`` — keep original labels unchanged.
- ``unified`` — remap labels through a user-supplied mapping dict
  onto the project's canonical ``UNIFIED_LABELS`` set.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.registry import UNIFIED_LABELS

logger = logging.getLogger(__name__)


def normalize_labels(
    labels: List[Any],
    label_names: List[str],
    mode: str = "per_dataset",
    unified_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[int], List[str]]:
    """Normalize a list of labels according to *mode*.

    Parameters
    ----------
    labels : list
        Raw labels — may be integer indices or string names.
    label_names : list[str]
        Original ordered label names (used to decode integer labels).
    mode : str
        ``"per_dataset"`` keeps original labels; ``"unified"`` remaps
        them via *unified_map*.
    unified_map : dict, optional
        Mapping ``{original_label_name: unified_label_name}``.
        Required when *mode* is ``"unified"``.

    Returns
    -------
    tuple[list[int], list[str]]
        ``(normalized_integer_labels, final_label_names)``
    """
    mode = mode.lower()

    if mode == "unified":
        return _normalize_unified(labels, label_names, unified_map or {})

    # ── per_dataset (default) ────────────────────────────────────────────
    return _normalize_per_dataset(labels, label_names)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalize_per_dataset(
    labels: List[Any],
    label_names: List[str],
) -> Tuple[List[int], List[str]]:
    """Pass-through: ensure labels are integer-encoded."""
    if not labels:
        return [], label_names

    # If labels are already integers, return directly
    if _all_integers(labels):
        return [int(l) for l in labels], list(label_names)

    # Build a name→id map from label_names
    name2id = {name: idx for idx, name in enumerate(label_names)}

    encoded: List[int] = []
    for lbl in labels:
        key = str(lbl)
        if key in name2id:
            encoded.append(name2id[key])
        else:
            # Auto-extend for unseen string labels
            new_idx = len(label_names)
            label_names = list(label_names) + [key]
            name2id[key] = new_idx
            encoded.append(new_idx)
            logger.warning(
                "Label '%s' not in label_names; auto-assigned index %d.",
                key,
                new_idx,
            )

    return encoded, list(label_names)


def _normalize_unified(
    labels: List[Any],
    label_names: List[str],
    unified_map: Dict[str, str],
) -> Tuple[List[int], List[str]]:
    """Remap labels to the unified label set and re-encode."""
    final_names = list(UNIFIED_LABELS)
    name2id = {name: idx for idx, name in enumerate(final_names)}

    encoded: List[int] = []
    for lbl in labels:
        # Decode integer labels to their original string name
        if isinstance(lbl, (int, np.integer)):
            if int(lbl) < len(label_names):
                original_name = label_names[int(lbl)]
            else:
                original_name = str(lbl)
        else:
            original_name = str(lbl)

        # Map to unified name
        unified_name = unified_map.get(original_name, original_name)

        if unified_name not in name2id:
            logger.warning(
                "Unified label '%s' (from '%s') not in UNIFIED_LABELS; "
                "skipping — assigning index 0.",
                unified_name,
                original_name,
            )
            encoded.append(0)
        else:
            encoded.append(name2id[unified_name])

    return encoded, final_names


def _all_integers(labels: List[Any]) -> bool:
    """Check if all elements are integer-like."""
    for lbl in labels:
        if not isinstance(lbl, (int, np.integer)):
            return False
    return True
