"""Evaluation metrics for classification tasks.

Provides functions to compute standard classification metrics (accuracy,
precision, recall, F1), confusion matrices, and formatted classification
reports.  All heavy lifting is delegated to scikit-learn.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    average : str
        Averaging strategy for multi-class metrics.  One of
        ``"micro"``, ``"macro"``, ``"weighted"``.

    Returns
    -------
    dict[str, float]
        ``{"accuracy": …, "precision": …, "recall": …, "f1": …}``
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        ),
    }

    logger.debug(
        "Metrics — Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return the confusion matrix as a NumPy array.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    np.ndarray of shape (n_classes, n_classes)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return confusion_matrix(y_true, y_pred)


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------
def format_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> str:
    """Return a formatted classification report string.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    label_names : list[str], optional
        Human-readable class names.

    Returns
    -------
    str
        Multi-line classification report produced by scikit-learn.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report: str = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
    )
    return report
