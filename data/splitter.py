"""Train / validation / test splitting utilities.

Provides stratified and random splitting for both pre-split HuggingFace
datasets and raw DataFrames, including low-resource sub-sampling.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def stratified_split(
    texts: List[str],
    labels: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Dict[str, list]]:
    """Split data into train / val / test with stratified sampling.

    Parameters
    ----------
    texts : list[str]
        Input texts.
    labels : list[int]
        Integer labels.
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"train": {"texts": [...], "labels": [...]}, "val": {...}, "test": {...}}``
    """
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    texts = np.array(texts)
    labels = np.array(labels)

    # First split: train vs (val+test)
    holdout_ratio = val_ratio + test_ratio
    try:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            texts, labels,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        logger.warning("First stratified split failed (likely due to sparse classes) — falling back to random split.")
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            texts, labels,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=None,
        )

    # Second split: val vs test
    relative_test = test_ratio / holdout_ratio
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout, y_holdout,
            test_size=relative_test,
            random_state=seed,
            stratify=y_holdout,
        )
    except ValueError:
        logger.warning("Second stratified split failed (likely due to sparse classes) — falling back to random split.")
        X_val, X_test, y_val, y_test = train_test_split(
            X_holdout, y_holdout,
            test_size=relative_test,
            random_state=seed,
            stratify=None,
        )

    splits = {
        "train": {"texts": X_train.tolist(), "labels": y_train.tolist()},
        "val":   {"texts": X_val.tolist(),   "labels": y_val.tolist()},
        "test":  {"texts": X_test.tolist(),  "labels": y_test.tolist()},
    }

    logger.info(
        f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    return splits


def subsample(
    texts: List[str],
    labels: List[int],
    fraction: float,
    stratified: bool = True,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """Sub-sample a dataset to a given fraction for low-resource experiments.

    Parameters
    ----------
    texts : list[str]
        Input texts.
    labels : list[int]
        Integer labels.
    fraction : float
        Fraction of data to keep (0.0, 1.0].
    stratified : bool
        If True, maintain label distribution.
    seed : int
        Random seed.

    Returns
    -------
    tuple[list[str], list[int]]
        Sub-sampled (texts, labels).
    """
    if fraction >= 1.0:
        return texts, labels

    n_total = len(texts)
    n_keep = max(1, int(n_total * fraction))

    rng = np.random.RandomState(seed)

    if stratified:
        labels_arr = np.array(labels)
        unique_labels = np.unique(labels_arr)
        keep_indices = []

        for lbl in unique_labels:
            lbl_indices = np.where(labels_arr == lbl)[0]
            n_lbl_keep = max(1, int(len(lbl_indices) * fraction))
            chosen = rng.choice(lbl_indices, size=n_lbl_keep, replace=False)
            keep_indices.extend(chosen.tolist())

        rng.shuffle(keep_indices)
    else:
        keep_indices = rng.choice(n_total, size=n_keep, replace=False).tolist()

    texts_sub = [texts[i] for i in keep_indices]
    labels_sub = [labels[i] for i in keep_indices]

    logger.info(
        f"Sub-sampled {fraction*100:.0f}%: {n_total} → {len(texts_sub)} samples"
    )
    return texts_sub, labels_sub


def get_label_distribution(labels: List[int]) -> Dict[int, int]:
    """Return a dict mapping label → count."""
    dist: Dict[int, int] = {}
    for lbl in labels:
        dist[lbl] = dist.get(lbl, 0) + 1
    return dict(sorted(dist.items()))


def print_split_stats(
    splits: Dict[str, Dict[str, list]],
    label_names: Optional[List[str]] = None,
) -> None:
    """Log statistics about a train/val/test split."""
    for split_name, data in splits.items():
        labels = data["labels"]
        dist = get_label_distribution(labels)
        logger.info(f"  {split_name:>5s}: {len(labels):>6d} samples, "
                     f"{len(dist)} classes")
        if label_names:
            for lbl_id, count in dist.items():
                name = label_names[lbl_id] if lbl_id < len(label_names) else f"#{lbl_id}"
                logger.info(f"         {name:>20s}: {count}")
