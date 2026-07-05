"""Support Vector Machine classifier for text classification.

Wraps ``sklearn.svm.LinearSVC`` behind the unified ``BaseClassifier``
interface.  Uses ``LinearSVC`` rather than ``SVC`` for efficiency on
high-dimensional text features.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.svm import LinearSVC

from models.classical.base import BaseClassifier

logger = logging.getLogger(__name__)


class SVMClassifier(BaseClassifier):
    """Linear SVM classifier.

    Configuration keys (from ``config.classical_models.svm``):
        - **kernel** (str): Kept for documentation; ``LinearSVC`` is always
          linear.  If ``"rbf"`` is specified the class logs a warning and
          falls back to linear.
        - **C** (float): Regularisation parameter.
        - **max_iter** (int): Maximum solver iterations.
    """

    def __init__(self, config: Any, num_labels: int, device: str = "cpu") -> None:
        super().__init__(config, num_labels, device)

        kernel: str = getattr(config, "kernel", "linear")
        c_val: float = getattr(config, "C", 1.0)
        max_iter: int = getattr(config, "max_iter", 5000)

        if kernel != "linear":
            logger.warning(
                "SVMClassifier uses LinearSVC; kernel='%s' is ignored "
                "(falling back to linear).",
                kernel,
            )

        self.model = LinearSVC(
            C=c_val,
            max_iter=max_iter,
            random_state=42,
            dual="auto",
        )
        logger.info(
            "LinearSVC initialised (C=%.4f, max_iter=%d)",
            c_val,
            max_iter,
        )

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fit the linear SVM model.

        Returns
        -------
        dict
            ``{"accuracy": float, "training_time": float}`` plus optional
            ``"val_accuracy"``.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        logger.info(
            "Training LinearSVC on %d samples (features=%d)",
            X_train.shape[0],
            X_train.shape[1],
        )

        start = time.perf_counter()
        self.model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        train_acc = float(self.model.score(X_train, y_train))
        history: Dict[str, Any] = {
            "accuracy": train_acc,
            "training_time": elapsed,
        }

        if X_val is not None and y_val is not None:
            val_acc = float(self.model.score(np.asarray(X_val), np.asarray(y_val)))
            history["val_accuracy"] = val_acc
            logger.info(
                "Training complete — train_acc=%.4f  val_acc=%.4f  time=%.2fs",
                train_acc,
                val_acc,
                elapsed,
            )
        else:
            logger.info(
                "Training complete — train_acc=%.4f  time=%.2fs",
                train_acc,
                elapsed,
            )

        return history

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X))

    # ------------------------------------------------------------------
    def get_params_count(self) -> int:
        """Number of fitted coefficients + intercepts."""
        if not hasattr(self.model, "coef_"):
            return 0
        n_params = int(self.model.coef_.size)
        if self.model.intercept_ is not None:
            n_params += int(np.asarray(self.model.intercept_).size)
        return n_params
