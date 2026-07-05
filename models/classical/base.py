"""Abstract base class for classical classifiers.

Every classical (non-quantum) model in the framework inherits from
``BaseClassifier`` and implements its ``train``, ``predict``, ``evaluate``,
and ``get_params_count`` methods.
"""

import abc
import logging
from typing import Any, Dict, Optional

import numpy as np

from evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class BaseClassifier(abc.ABC):
    """Base interface shared by all classical classifiers.

    Parameters
    ----------
    config : Config
        Model-specific configuration block (e.g. ``config.classical_models.svm``).
    num_labels : int
        Number of target classes.
    device : str
        ``"cpu"`` or ``"cuda"`` — relevant only for PyTorch-backed models.
    """

    def __init__(self, config: Any, num_labels: int, device: str = "cpu") -> None:
        self.config = config
        self.num_labels = num_labels
        self.device = device
        self.model: Any = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fit the model and return a training-history dict.

        Returns
        -------
        dict
            At minimum ``{"training_time": float}``.  PyTorch models also
            return per-epoch ``train_loss``, ``val_loss``, ``train_acc``,
            ``val_acc`` lists.
        """

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices for input features *X*.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, feature_dim)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """

    @abc.abstractmethod
    def get_params_count(self) -> int:
        """Return the total number of (trainable) model parameters."""

    # ------------------------------------------------------------------
    # Shared convenience
    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Predict on *X* and compute metrics against *y*.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, feature_dim)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        dict[str, float]
            ``{"accuracy": …, "precision": …, "recall": …, "f1": …}``
        """
        y_pred = self.predict(X)
        metrics = compute_metrics(np.asarray(y), y_pred)
        logger.info(
            "Evaluation — Acc=%.4f  P=%.4f  R=%.4f  F1=%.4f",
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
        return metrics
