"""PyTorch CNN classifier for text classification.

Implements ``TextCNN`` (multi-kernel Conv1d + max-over-time pooling) and
``CNNClassifier``, a ``BaseClassifier`` wrapper that handles numpy ↔ tensor
conversion, DataLoader batching, and per-epoch logging.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from evaluation.metrics import compute_metrics
from models.classical.base import BaseClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch module
# ---------------------------------------------------------------------------
class TextCNN(nn.Module):
    """Multi-kernel 1-D CNN for text / embedding classification.

    Input shape : ``(batch, 1, feature_dim)``
    Output shape: ``(batch, num_labels)``

    Architecture
    ------------
    For each kernel size *k*:
        Conv1d(1, num_filters, k) → ReLU → AdaptiveMaxPool1d(1)
    Concatenate all filter outputs → Dropout → Linear → logits
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        num_filters: int = 128,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        kernel_sizes = kernel_sizes or [3, 4, 5]

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,  # keep length roughly the same
                )
                for k in kernel_sizes
            ]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 1, feature_dim)
        """
        conv_outs: List[torch.Tensor] = []
        for conv in self.convs:
            c = self.relu(conv(x))                       # (B, F, L')
            c = torch.adaptive_max_pool1d(c, 1)[0].squeeze(-1)  # (B, F)
            conv_outs.append(c)

        out = torch.cat(conv_outs, dim=1)  # (B, F * n_kernels)
        out = self.dropout(out)
        logits = self.fc(out)              # (B, num_labels)
        return logits


# ---------------------------------------------------------------------------
# BaseClassifier wrapper
# ---------------------------------------------------------------------------
class CNNClassifier(BaseClassifier):
    """Wrapper that trains a ``TextCNN`` on pre-computed embedding arrays.

    Configuration keys (from ``config.classical_models.cnn``):
        - **num_filters** (int)
        - **kernel_sizes** (list[int])
        - **dropout** (float)
        - **epochs** (int)
        - **batch_size** (int)
        - **learning_rate** (float)
    """

    def __init__(self, config: Any, num_labels: int, device: str = "cpu") -> None:
        super().__init__(config, num_labels, device)

        self.num_filters: int = getattr(config, "num_filters", 128)
        self.kernel_sizes: List[int] = list(getattr(config, "kernel_sizes", [3, 4, 5]))
        self.dropout: float = getattr(config, "dropout", 0.5)
        self.epochs: int = getattr(config, "epochs", 20)
        self.batch_size: int = getattr(config, "batch_size", 64)
        self.lr: float = getattr(config, "learning_rate", 0.001)

        # The actual nn.Module is built lazily in train() once we know input_dim
        self._net: Optional[TextCNN] = None
        logger.info(
            "CNNClassifier initialised (filters=%d, kernels=%s, dropout=%.2f, "
            "epochs=%d, bs=%d, lr=%.4f)",
            self.num_filters,
            self.kernel_sizes,
            self.dropout,
            self.epochs,
            self.batch_size,
            self.lr,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_net(self, input_dim: int) -> TextCNN:
        net = TextCNN(
            input_dim=input_dim,
            num_labels=self.num_labels,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            dropout=self.dropout,
        )
        return net.to(self.device)

    @staticmethod
    def _to_loader(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Convert numpy arrays to a ``DataLoader``.

        Reshapes X from (n, d) → (n, 1, d) for Conv1d.
        """
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (n,1,d)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop.

        Returns
        -------
        dict
            ``{"train_loss": [...], "val_loss": [...], "train_acc": [...],
            "val_acc": [...], "training_time": float}``
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        input_dim = X_train.shape[1]

        self._net = self._build_net(input_dim)
        self.model = self._net  # expose for get_params_count

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = self._to_loader(X_train, y_train, self.batch_size, shuffle=True)
        val_loader: Optional[DataLoader] = None
        if X_val is not None and y_val is not None:
            val_loader = self._to_loader(
                np.asarray(X_val, dtype=np.float32),
                np.asarray(y_val, dtype=np.int64),
                self.batch_size,
                shuffle=False,
            )

        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "training_time": 0.0,
        }

        logger.info(
            "Starting CNN training — %d epochs, %d train samples, input_dim=%d",
            self.epochs,
            len(X_train),
            input_dim,
        )

        start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            # --- train ---
            self._net.train()
            running_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{self.epochs}",
                leave=False,
            )
            for xb, yb in pbar:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits = self._net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
                pbar.set_postfix(loss=loss.item(), acc=correct / total)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(epoch_acc)

            # --- validation ---
            if val_loader is not None:
                val_loss, val_acc = self._eval_loader(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f  train_acc=%.4f  "
                    "val_loss=%.4f  val_acc=%.4f",
                    epoch,
                    self.epochs,
                    epoch_loss,
                    epoch_acc,
                    val_loss,
                    val_acc,
                )
            else:
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f  train_acc=%.4f",
                    epoch,
                    self.epochs,
                    epoch_loss,
                    epoch_acc,
                )

        history["training_time"] = time.perf_counter() - start
        logger.info("CNN training finished in %.2fs", history["training_time"])
        return history

    # ------------------------------------------------------------------
    def _eval_loader(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple:
        """Evaluate on a DataLoader, return ``(loss, accuracy)``."""
        self._net.eval()  # type: ignore[union-attr]
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self._net(xb)  # type: ignore[union-attr]
                loss = criterion(logits, yb)

                running_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)

        return running_loss / total, correct / total

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model has not been trained yet.")

        self._net.eval()
        X_t = torch.tensor(
            np.asarray(X, dtype=np.float32), dtype=torch.float32
        ).unsqueeze(1).to(self.device)

        all_preds: List[np.ndarray] = []
        with torch.no_grad():
            # process in batches to avoid OOM
            for i in range(0, X_t.size(0), self.batch_size):
                batch = X_t[i : i + self.batch_size]
                logits = self._net(batch)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())

        return np.concatenate(all_preds)

    # ------------------------------------------------------------------
    def get_params_count(self) -> int:
        if self._net is None:
            return 0
        return sum(p.numel() for p in self._net.parameters())
