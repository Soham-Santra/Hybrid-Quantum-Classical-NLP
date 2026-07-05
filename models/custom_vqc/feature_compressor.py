"""Feature compression from embedding dimension to qubit count.

Maps high-dimensional classical feature vectors (e.g. TF-IDF, BERT embeddings)
down to a small number of features equal to the qubit count so they can be
angle-encoded into a quantum circuit.

This is a local duplicate of the quantum_template version so the custom_vqc
sub-package remains self-contained.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureCompressor(nn.Module):
    """Linear projection followed by an activation to compress features.

    The default ``tanh`` activation scales outputs to ``(-1, 1)`` which,
    when multiplied by π downstream, maps naturally to ``(-π, π)`` —
    a suitable range for angle-encoding gates such as ``RY`` / ``RZ``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vectors.
    n_qubits : int
        Number of qubits (= output dimensionality).
    activation : str, optional
        Activation function applied after the linear layer.
        Supported: ``"tanh"`` (default), ``"sigmoid"``, ``"none"``.
    """

    _ACTIVATIONS = {
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "none": nn.Identity,
    }

    def __init__(
        self,
        input_dim: int,
        n_qubits: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()

        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {list(self._ACTIVATIONS.keys())}."
            )

        self.linear = nn.Linear(input_dim, n_qubits)
        self.activation = self._ACTIVATIONS[activation]()

        logger.info(
            "FeatureCompressor: %d → %d (activation=%s)",
            input_dim,
            n_qubits,
            activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input features to qubit-sized representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Compressed tensor of shape ``(batch_size, n_qubits)``.
        """
        return self.activation(self.linear(x))
