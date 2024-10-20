"""
This module provides a basic framework for creating quantum circuits and
applying quantum gate operations using a simulator. It supports methods
to initialize circuits and perform operations such as Hadamard (H),
Pauli-X, and controlled-NOT (CNOT) gates.
"""

from abc import ABC
from enum import Enum
from typing import Any, Callable

import numpy as np

from simulator.matrix_simulator import MatrixSimulator
from simulator.simulator import Simulator
from simulator.tensor_simulator import TensorSimulator

# Hadamard gate
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

# X gate (Pauli-X)
X = np.array([[0, 1], [1, 0]])

""" 
 |control⟩ ----●---- 
               |
 |target⟩  ----X----
"""
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


class QuantumCircuit(ABC):
    """
    A quantum circuit representation that uses simulator to apply gates.
    Supports basic quantum gate operations such as H, X, and CNOT
    """

    def __init__(self, simulator: Simulator):
        """
        Initializes quantum circuit, with simulator to apply gates.
        """
        self.simulator = simulator

    def get_state(self) -> np.ndarray:
        """
        Get the current state vector of the circuit.
        """
        return self.simulator.get_state()

    def h(self, qubit: int):
        """
        Apply the Hadamard gate to a specified qubit.
        """
        self.simulator.apply_single_gate(H, qubit)

    def x(self, qubit: int):
        """
        Apply the Pauli-X gate (NOT gate) to a specified qubit.
        """
        self.simulator.apply_single_gate(X, qubit)

    def cnot(self, control_qubit: int, target_qubit: int):
        """
        Apply a controlled-NOT (CNOT) gate between two qubits.
        """
        if control_qubit == target_qubit:
            raise ValueError("Control qubit cannot be the target qubit")

        self.simulator.apply_control_gate(CNOT, control_qubit, target_qubit)


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
    state = np.zeros(2**num_qubits)
    state[0] = 1
    return state


def __get_simulator_type(simulator_type: SimulatorType) -> Callable[[Any], Simulator]:
    if simulator_type is SimulatorType.MATRIX:
        return lambda state: MatrixSimulator(state=state)
    elif simulator_type is SimulatorType.TENSOR:
        return lambda state: TensorSimulator(state=state)
    else:
        raise ValueError(f"Unsupported simulator type: {simulator_type}")
