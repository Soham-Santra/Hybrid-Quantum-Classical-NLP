"""Custom VQC v2 — Data re-uploading variational quantum circuit.

Same structure as v1 but with **data re-uploading**: the input features are
re-encoded via RY gates at the beginning of *every* variational layer rather
than only once.  This strategy has been shown to increase the expressivity of
shallow circuits (Pérez-Salinas et al., 2020).
"""

import logging
from typing import Any, Dict, Tuple

import pennylane as qml

logger = logging.getLogger(__name__)

WeightShapes = Dict[str, Tuple[int, ...]]


def create_custom_vqc_v2(
    n_qubits: int,
    n_layers: int,
) -> Tuple[qml.QNode, WeightShapes]:
    """Create a data re-uploading VQC circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.

    Returns
    -------
    tuple[qml.QNode, dict]
        The quantum circuit and its weight shapes.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs: Any, weights: Any) -> list:
        for layer in range(n_layers):
            # Re-upload data at every layer
            for i in range(n_qubits):
                qml.RY(inputs[..., i], wires=i)

            # Variational rotations
            for qubit in range(n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)

            # Linear entanglement
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes: WeightShapes = {"weights": (n_layers, n_qubits, 3)}

    logger.info(
        "Created Custom VQC v2 (data re-uploading): %d qubits, %d layers",
        n_qubits,
        n_layers,
    )
    return circuit, weight_shapes
