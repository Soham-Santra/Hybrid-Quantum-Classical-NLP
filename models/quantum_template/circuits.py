"""PennyLane template-based quantum circuits for hybrid classification.

Provides factory functions that return ``(qnode, weight_shapes)`` pairs
ready to be wrapped by ``qml.qnn.TorchLayer``.

Available circuit types
-----------------------
- ``strongly_entangling`` — ``qml.StronglyEntanglingLayers``
- ``basic_entangler``     — ``qml.BasicEntanglerLayers``
- ``efficient_su2``       — hand-built RY+RZ rotations with linear CNOT chain
"""

import logging
from typing import Any, Callable, Dict, Tuple

import pennylane as qml

logger = logging.getLogger(__name__)

# Type aliases for readability
WeightShapes = Dict[str, Tuple[int, ...]]
CircuitFactory = Callable[..., Tuple[qml.QNode, WeightShapes]]


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def create_strongly_entangling_circuit(
    n_qubits: int,
    n_layers: int,
) -> Tuple[qml.QNode, WeightShapes]:
    """Strongly entangling layers with angle embedding.

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
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes: WeightShapes = {"weights": (n_layers, n_qubits, 3)}

    logger.info(
        "Created StronglyEntangling circuit: %d qubits, %d layers",
        n_qubits,
        n_layers,
    )
    return circuit, weight_shapes


def create_basic_entangler_circuit(
    n_qubits: int,
    n_layers: int,
) -> Tuple[qml.QNode, WeightShapes]:
    """Basic entangler layers with angle embedding.

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
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes: WeightShapes = {"weights": (n_layers, n_qubits)}

    logger.info(
        "Created BasicEntangler circuit: %d qubits, %d layers",
        n_qubits,
        n_layers,
    )
    return circuit, weight_shapes


def create_efficient_su2_circuit(
    n_qubits: int,
    n_layers: int,
) -> Tuple[qml.QNode, WeightShapes]:
    """Manually-built efficient SU(2) ansatz: RY+RZ rotations with linear CNOT chain.

    This does **not** use any PennyLane template — the circuit is assembled
    gate-by-gate for full transparency and control.

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
        # Data encoding
        qml.AngleEmbedding(inputs, wires=range(n_qubits))

        # Variational layers
        for layer in range(n_layers):
            # RY + RZ rotations on each qubit
            for qubit in range(n_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)

            # Linear CNOT entanglement chain
            for qubit in range(n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes: WeightShapes = {"weights": (n_layers, n_qubits, 2)}

    logger.info(
        "Created EfficientSU2 circuit: %d qubits, %d layers",
        n_qubits,
        n_layers,
    )
    return circuit, weight_shapes


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CIRCUIT_REGISTRY: Dict[str, CircuitFactory] = {
    "strongly_entangling": create_strongly_entangling_circuit,
    "basic_entangler": create_basic_entangler_circuit,
    "efficient_su2": create_efficient_su2_circuit,
}


def get_circuit(
    circuit_type: str,
    n_qubits: int,
    n_layers: int,
) -> Tuple[qml.QNode, WeightShapes]:
    """Select and instantiate a circuit by name.

    Parameters
    ----------
    circuit_type : str
        One of ``"strongly_entangling"``, ``"basic_entangler"``,
        ``"efficient_su2"``.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.

    Returns
    -------
    tuple[qml.QNode, dict]
        The quantum circuit and its weight shapes.

    Raises
    ------
    ValueError
        If *circuit_type* is not recognised.
    """
    if circuit_type not in _CIRCUIT_REGISTRY:
        raise ValueError(
            f"Unknown circuit type '{circuit_type}'. "
            f"Available: {list(_CIRCUIT_REGISTRY.keys())}"
        )

    builder = _CIRCUIT_REGISTRY[circuit_type]
    return builder(n_qubits, n_layers)
