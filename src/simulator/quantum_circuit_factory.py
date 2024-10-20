from enum import Enum
from typing import Any, Callable

import numpy as np

from simulator import QuantumCircuit
from simulator.matrix_simulator import MatrixSimulator
from simulator.simulator import Simulator
from simulator.tensor_simulator import TensorSimulator

SimulatorType = Enum("SimulatorType", ["MATRIX", "TENSOR"])


def create_quantum_circuit_from_qubits(
    simulator_type: SimulatorType, num_qubits: int
) -> QuantumCircuit:
    """
    Create a circuit with the specified number of qubits, initialized to the zero state.
    """
    state = zero_state(num_qubits)
    simulator = __get_simulator_type(simulator_type)
    return QuantumCircuit(simulator(state))


def create_quantum_circuit_from_state(
    simulator_type: SimulatorType, state: np.ndarray
) -> QuantumCircuit:
    """
    Create a circuit initialized to a given state.
    """
    simulator = __get_simulator_type(simulator_type)
    return QuantumCircuit(simulator(state))


def zero_state(num_qubits: int) -> np.ndarray:
    """Returns state when all qubits are in state |0>"""

    if num_qubits < 1:
        raise ValueError("num_qubits must be greater than or equal to 1.")
    state = np.zeros(2**num_qubits, dtype=np.complex128)
    state[0] = 1
    return state


def __get_simulator_type(simulator_type: SimulatorType) -> Callable[[Any], Simulator]:
    if simulator_type is SimulatorType.MATRIX:
        return lambda state: MatrixSimulator(state=state)
    elif simulator_type is SimulatorType.TENSOR:
        return lambda state: TensorSimulator(state=state)
    else:
        raise ValueError(f"Unsupported simulator type: {simulator_type}")
