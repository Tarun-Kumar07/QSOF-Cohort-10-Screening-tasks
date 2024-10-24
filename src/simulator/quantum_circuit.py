"""
This module provides a basic framework for creating quantum circuits and
applying quantum gate operations using a simulator. It supports methods
to initialize circuits and perform operations such as Hadamard (H),
Pauli-X, and controlled-NOT (CNOT) gates.
"""

from abc import ABC
from typing import Dict

import numpy as np

from simulator.simulator import Simulator

# Hadamard gate
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

# X gate (Pauli-X)
X = np.array([[0, 1], [1, 0]])

# Y gate (Pauli-Y)
Y = np.array([[0, -1j], [1j, 0]])

# Z gate (Pauli-Z)
Z = np.array([[1, 0], [0, -1]])

PAULIS = ["X", "Y", "Z"]

""" 
 |control⟩ ----●---- 
               |
 |target⟩  ----X----
"""
CNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


class QuantumCircuit(ABC):
    """
    A quantum circuit representation that uses simulator to apply gates.
    Supports basic quantum gate operations such as H, X, and CNOT
    """

    def __init__(self, simulator: Simulator, name: str = None):
        """
        Initializes quantum circuit, with simulator to apply gates.
        """
        self.simulator = simulator
        self.name = name

    def get_state(self) -> np.ndarray:
        """
        Get the current state vector of the circuit.
        """
        return self.simulator.get_state()

    def get_num_qubits(self) -> int:
        """
        Get the current state vector of the circuit.
        """
        return self.simulator.num_qubits

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

    def y(self, qubit: int):
        """
        Apply the Pauli-Y gate (NOT gate) to a specified qubit.
        """
        self.simulator.apply_single_gate(Y, qubit)

    def z(self, qubit: int):
        """
        Apply the Pauli-Y gate (NOT gate) to a specified qubit.
        """
        self.simulator.apply_single_gate(Z, qubit)

    def cnot(self, control_qubit: int, target_qubit: int):
        """
        Apply a controlled-NOT (CNOT) gate between two qubits.
        """
        if control_qubit == target_qubit:
            raise ValueError("Control qubit cannot be the target qubit")

        self.simulator.apply_control_gate(CNOT, control_qubit, target_qubit)


def sample(qc: QuantumCircuit, sample_count: int) -> Dict[str, int]:
    if sample_count < 0:
        raise ValueError("Sample count cannot be negative")

    state_vector = qc.get_state()
    probs = np.abs(state_vector) ** 2

    num_qubits = qc.get_num_qubits()
    possible_states = np.arange(2**num_qubits)
    possible_bit_strings = [
        format(state, f"0{num_qubits}b") for state in possible_states
    ]

    sampled_states = np.random.choice(possible_bit_strings, size=sample_count, p=probs)

    samples = {}
    for state in sampled_states:
        samples[state] = samples.get(state, 0) + 1

    return samples


def expectation(qc: QuantumCircuit, pauli_word: Dict[int, str]) -> float:
    state_before_applying_observables = qc.get_state()
    for qubit, pauli in pauli_word.items():
        if pauli == "X":
            qc.x(qubit)
        elif pauli == "Y":
            qc.y(qubit)
        elif pauli == "Z":
            qc.z(qubit)
        else:
            raise ValueError(
                f"For qubit {qubit}, '{pauli}' is not one of pauli observables"
            )

    exp = np.dot(state_before_applying_observables, qc.get_state())

    return exp
