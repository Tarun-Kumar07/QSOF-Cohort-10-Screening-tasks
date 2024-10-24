"""
This module defines an abstract base class for quantum simulators and
specifies methods for applying single and controlled quantum gates.
"""

from abc import ABC, abstractmethod

import numpy as np


def is_power_of_2(n) -> bool:
    """Check if n is a power of 2."""
    return (n & (n - 1)) == 0


def validate_state(state: np.ndarray):
    """Validates provided state is normalized and of right length"""
    if not np.allclose(np.linalg.norm(state), 1):
        raise ValueError("state must be normalized")

    len_state = len(state)
    if len_state < 2 or not is_power_of_2(len_state):
        raise ValueError("len(state) must be a power of 2")


class Simulator(ABC):
    """
    Abstract base class for a quantum simulator.

    Provides methods for simulating quantum states and applying quantum gates.
    This class contains logic for validating state and qubit indices to apply gates
    """

    def __init__(self, state: np.ndarray):
        """
        Initialize the simulator with the given quantum state.
        """
        validate_state(state)

        self.state = state
        self.num_qubits = int(np.log2(len(state)))

    def get_state(self) -> np.ndarray:
        """
        Return the current quantum state.
        """
        return self.state

    def apply_single_gate(self, gate: np.ndarray, qubit: int):
        """
        Validates qubit and applies a single-qubit gate.
        """
        self.validate_qubit(qubit)
        self._apply_single_gate(gate, qubit)

    def apply_control_gate(self, gate: np.ndarray, control: int, target: int):
        """
        Validates qubits and applies a controlled gate.
        """
        self.validate_qubit(control)
        self.validate_qubit(target)

        if control == target:
            raise ValueError("Control qubit cannot be the target qubit")

        self._apply_control_gate(gate, control, target)

    def validate_qubit(self, qubit: int):
        """
        Validate the qubit index.
        """
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"qubit must be in [0, {self.num_qubits - 1}]")

    @abstractmethod
    def _apply_single_gate(self, gate: np.ndarray, qubit: int):
        """
        Abstract method to apply a single-qubit gate. Must be implemented
        by subclasses.
        """

    @abstractmethod
    def _apply_control_gate(
        self, gate: np.ndarray, control_qubit: int, target_qubit: int
    ):
        """
        Abstract method to apply a controlled gate. Must be implemented
        by subclasses.
        """
