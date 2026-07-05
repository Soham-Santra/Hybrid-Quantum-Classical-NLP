"""Logistic Regression classifier for text classification.

Wraps ``sklearn.linear_model.LogisticRegression`` behind the unified
``BaseClassifier`` interface used throughout the framework.
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from models.classical.base import BaseClassifier

logger = logging.getLogger(__name__)


class LogisticRegressionClassifier(BaseClassifier):
    """Logistic Regression classifier.

    Configuration keys (from ``config.classical_models.logistic_regression``):
        - **C** (float): Inverse regularisation strength.
        - **max_iter** (int): Maximum solver iterations.
        - **solver** (str): Optimisation algorithm.
        - **multi_class** (str): Multi-class strategy.
    """

    def __init__(self, config: Any, num_labels: int, device: str = "cpu") -> None:
        super().__init__(config, num_labels, device)

        c_val: float = getattr(config, "C", 1.0)
        max_iter: int = getattr(config, "max_iter", 1000)
        solver: str = getattr(config, "solver", "lbfgs")
        multi_class: str = getattr(config, "multi_class", "multinomial")

        self.model = LogisticRegression(
            C=c_val,
            max_iter=max_iter,
            solver=solver,
            random_state=42,
            n_jobs=-1,
        )
        logger.info(
            "LogisticRegression initialised (C=%.4f, solver=%s, max_iter=%d)",
            c_val,
            solver,
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
        """Fit the logistic regression model.

        Returns
        -------
        dict
            ``{"accuracy": float, "training_time": float}`` plus optional
            ``"val_accuracy"`` when validation data is provided.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        logger.info(
            "Training LogisticRegression on %d samples (features=%d)",
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
