"""
This module implements a concrete quantum simulator using tensor
operations, providing methods to manipulate quantum states with gate
operations.
"""

import numpy as np

from simulator.simulator import Simulator


class TensorSimulator(Simulator):
    """
    A concrete quantum simulator that uses tensor operations to simulate
    quantum gate applications on quantum states.
    """

    def __init__(self, state: np.ndarray):
        super().__init__(state)
        new_shape = (2,) * self.num_qubits
        self.state = state.reshape(new_shape)

    def _apply_single_gate(self, gate: np.ndarray, qubit: int):
        big_endian_qubit_index = self.convert_to_big_endian_qubit(qubit)
        next_state = np.tensordot(gate, self.state, (1, big_endian_qubit_index))
        self.state = np.moveaxis(next_state, 0, big_endian_qubit_index)

    def _apply_control_gate(self, gate: np.ndarray, control_qubit: int, target_qubit: int):
        gate_tensor = gate.reshape(2, 2, 2, 2)
        big_endian_control = self.convert_to_big_endian_qubit(control_qubit)
        big_endian_target = self.convert_to_big_endian_qubit(target_qubit)
        next_state = np.tensordot(
            gate_tensor, self.state, ((2, 3), (big_endian_target, big_endian_control))
        )
        self.state = np.moveaxis(next_state, (0, 1), (big_endian_target, big_endian_control))

    def convert_to_big_endian_qubit(self, qubit):
        """
        This is required as tensor product in the article assumed qubits to be in big endian order.
        """
        return self.num_qubits - qubit - 1

    def get_state(self) -> np.ndarray:
        """
        Return the current quantum state in flattened form.
        """
        return self.state.reshape(2**self.num_qubits)
