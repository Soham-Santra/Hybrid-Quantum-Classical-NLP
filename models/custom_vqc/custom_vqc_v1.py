"""Custom VQC v1 — Basic variational quantum circuit.

A hand-built variational circuit (no PennyLane templates) with:

- **Data encoding**: RY angle embedding (applied once at the start).
- **Variational layers**: RX → RY → RZ rotations on each qubit.
- **Entanglement**: Linear CNOT chain (qubit 0→1→2→…→n-1).
"""

import logging
from typing import Any, Dict, Tuple

import pennylane as qml

logger = logging.getLogger(__name__)

WeightShapes = Dict[str, Tuple[int, ...]]


def create_custom_vqc_v1(
    n_qubits: int,
    n_layers: int,
) -> Tuple[qml.QNode, WeightShapes]:
    """Create a basic custom VQC circuit.

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
        # Data encoding: RY angle embedding (once)
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)

        # Variational layers
        for layer in range(n_layers):
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
        "Created Custom VQC v1 (basic): %d qubits, %d layers",
        n_qubits,
        n_layers,
    )
    return circuit, weight_shapes
