"""End-to-end hybrid quantum-classical model using custom VQC circuits.

Architecture::

    Input features → FeatureCompressor → TorchLayer(custom VQC) → Linear → logits

Provides both a raw ``nn.Module`` (``CustomVQCModel``) and a high-level
``CustomVQCClassifier`` with ``train`` / ``predict`` / ``evaluate`` API.

The VQC version is selected via ``config.custom_vqc.version``:

- ``"v1"`` — basic (single data encoding, linear entanglement)
- ``"v2"`` — data re-uploading
- ``"v3"`` — ring entanglement
"""

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import pennylane as qml

from models.custom_vqc.feature_compressor import FeatureCompressor
from models.custom_vqc.custom_vqc_v1 import create_custom_vqc_v1
from models.custom_vqc.custom_vqc_v2 import create_custom_vqc_v2
from models.custom_vqc.custom_vqc_v3 import create_custom_vqc_v3

logger = logging.getLogger(__name__)

# Registry mapping version strings to factory functions
_VQC_VERSIONS: Dict[str, Callable] = {
    "v1": create_custom_vqc_v1,
    "v2": create_custom_vqc_v2,
    "v3": create_custom_vqc_v3,
}


def _get_vqc_circuit(
    version: str,
    n_qubits: int,
    n_layers: int,
) -> Tuple[qml.QNode, Dict[str, Tuple[int, ...]]]:
    """Select a custom VQC circuit by version string.

    Raises
    ------
    ValueError
        If the requested version is not recognised.
    """
    if version not in _VQC_VERSIONS:
        raise ValueError(
            f"Unknown custom VQC version '{version}'. "
            f"Available: {list(_VQC_VERSIONS.keys())}"
        )
    return _VQC_VERSIONS[version](n_qubits, n_layers)


# ---------------------------------------------------------------------------
# nn.Module — hybrid network
# ---------------------------------------------------------------------------

class CustomVQCModel(nn.Module):
    """Hybrid quantum-classical model backed by a custom VQC circuit.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vectors.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    num_classes : int
        Number of output classes.
    version : str
        Custom VQC version: ``"v1"``, ``"v2"``, or ``"v3"``.
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int,
        n_layers: int,
        num_classes: int,
        version: str = "v1",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.version = version

        # 1. Classical feature compressor
        self.compressor = FeatureCompressor(input_dim, n_qubits, activation="tanh")

        # 2. Quantum circuit wrapped as a PyTorch layer
        circuit, weight_shapes = _get_vqc_circuit(version, n_qubits, n_layers)
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # 3. Classical output head
        self.classifier = nn.Linear(n_qubits, num_classes)

        logger.info(
            "CustomVQCModel: input=%d, qubits=%d, layers=%d, "
            "classes=%d, version=%s",
            input_dim,
            n_qubits,
            n_layers,
            num_classes,
            version,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compress → quantum → classify.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch_size, num_classes)``.
        """
        x = self.compressor(x)          # (B, n_qubits)
        x = self.quantum_layer(x)       # (B, n_qubits)
        x = self.classifier(x)          # (B, num_classes)
        return x


# ---------------------------------------------------------------------------
# High-level classifier wrapper
# ---------------------------------------------------------------------------

def _compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy from logits and integer labels."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


class CustomVQCClassifier:
    """High-level wrapper for training and evaluating a ``CustomVQCModel``.

    Parameters
    ----------
    config : Config
        Framework configuration object.  Reads from ``config.custom_vqc``.
    num_labels : int
        Number of output classes.
    input_dim : int
        Dimensionality of input feature vectors.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        config: Any,
        num_labels: int,
        input_dim: int,
        device: str = "cpu",
    ) -> None:
        vqc_cfg = config.custom_vqc

        self.n_qubits: int = getattr(vqc_cfg, "n_qubits", 4)
        self.n_layers: int = getattr(vqc_cfg, "n_layers", 2)
        self.version: str = getattr(vqc_cfg, "version", "v1")
        self.epochs: int = getattr(vqc_cfg, "epochs", 30)
        self.batch_size: int = getattr(vqc_cfg, "batch_size", 32)
        self.lr: float = getattr(vqc_cfg, "learning_rate", 0.01)
        self.device = torch.device(device)
        self.num_labels = num_labels
        self.input_dim = input_dim

        self.model = CustomVQCModel(
            input_dim=input_dim,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            num_classes=num_labels,
            version=self.version,
        ).to(self.device)

        logger.info(
            "CustomVQCClassifier initialised on %s "
            "(qubits=%d, layers=%d, version=%s)",
            self.device,
            self.n_qubits,
            self.n_layers,
            self.version,
        )

    # ------------------------------------------------------------------ train
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Train the hybrid model with Adam + CrossEntropyLoss.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training features and integer labels.
        X_val, y_val : np.ndarray
            Validation features and integer labels.

        Returns
        -------
        dict
            Training history with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc``, ``training_time``.
        """
        # Prepare data loaders
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        start_time = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            # --- Training ---
            self.model.train()
            running_loss = 0.0
            running_acc = 0.0
            n_batches = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{self.epochs}",
                leave=False,
            )
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_acc += _compute_accuracy(logits, batch_y)
                n_batches += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / max(n_batches, 1)
            train_acc = running_acc / max(n_batches, 1)

            # --- Validation ---
            val_loss, val_acc = self._evaluate_loader(val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f  train_acc=%.4f  "
                "val_loss=%.4f  val_acc=%.4f",
                epoch,
                self.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

        elapsed = time.perf_counter() - start_time
        history["training_time"] = elapsed  # type: ignore[assignment]
        logger.info("Training completed in %.2f seconds.", elapsed)
        return history

    # --------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, input_dim)``.

        Returns
        -------
        np.ndarray
            Predicted labels of shape ``(n_samples,)``.
        """
        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        all_preds: list = []

        with torch.no_grad():
            loader = DataLoader(
                TensorDataset(tensor_x),
                batch_size=self.batch_size,
                shuffle=False,
            )
            for (batch_x,) in loader:
                logits = self.model(batch_x)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)

        return np.concatenate(all_preds)

    # -------------------------------------------------------------- evaluate
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model and return metric scores.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Ground-truth integer labels.

        Returns
        -------
        dict[str, float]
            Dictionary with ``accuracy``, ``precision``, ``recall``, ``f1``.
        """
        from evaluation.metrics import compute_metrics

        preds = self.predict(X)
        return compute_metrics(y, preds)

    # --------------------------------------------------------- internal utils
    def _evaluate_loader(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Run the model on a ``DataLoader`` and return (loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item()
                total_acc += _compute_accuracy(logits, batch_y)
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)
        return avg_loss, avg_acc

    def get_params_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
