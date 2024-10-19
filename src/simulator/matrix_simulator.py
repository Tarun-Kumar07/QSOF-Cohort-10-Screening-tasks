"""
This module implements a concrete quantum simulator using matrix
operations, providing methods to manipulate quantum states with gate
operations.
"""

import numpy as np

from simulator.simulator import Simulator

I = np.eye(2)

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


class MatrixSimulator(Simulator):
    """
    A concrete quantum simulator that uses matrix operations to simulate
    quantum gate applications on quantum states.
    """

    def _apply_single_gate(self, gate: np.ndarray, qubit: int):
        unitary = np.eye(1)
        for q in range(self.num_qubits):
            q_gate = gate if q == qubit else I
            unitary = np.kron(unitary, q_gate)

        self.state = np.dot(unitary, self.state)

    def _apply_control_gate(
        self, gate: np.ndarray, control_qubit: int, target_qubit: int
    ):
        if control_qubit == target_qubit:
            raise ValueError("Control qubit cannot be the target qubit")

        # control qubit and target qubit are far away
        if control_qubit < target_qubit:
            # bring control qubit closer to target qubit using swap gates
            for q in range(control_qubit, target_qubit - 1):
                self.__swap_with_neighbours(q, q + 1)
            # control qubit is just above target qubit
            self.__apply_double_qubit_gate_with_next_qubit(gate, target_qubit - 1)
            for q in range(target_qubit - 1, control_qubit, -1):
                self.__swap_with_neighbours(q, q - 1)
        else:
            # bring target qubit closer to control qubit using swap gates
            # note there is an extra swap to take target qubit below control qubit
            for q in range(target_qubit, control_qubit):
                self.__swap_with_neighbours(q, q + 1)
            # control qubit is now just above provided value
            self.__apply_double_qubit_gate_with_next_qubit(gate, control_qubit - 1)
            for q in range(control_qubit, target_qubit, -1):
                self.__swap_with_neighbours(q, q - 1)

    def __swap_with_neighbours(self, qubit1, qubit2):
        """
        Apply the SWAP gate between two neighboring qubits.

        Parameters:
            qubit1 (int): The first qubit index.
            qubit2 (int): The second qubit index.
        """
        if abs(qubit1 - qubit2) != 1:
            raise ValueError(
                "The SWAP gate can only be applied between neighboring qubits."
            )

        # Determine the smaller and larger indices
        min_qubit = min(qubit1, qubit2)
        self.__apply_double_qubit_gate_with_next_qubit(SWAP, min_qubit)

    def __apply_double_qubit_gate_with_next_qubit(self, gate: np.ndarray, qubit: int):
        """
        Apply a two-qubit gate between a specified qubit and its next qubit.

        Parameters:
            gate (np.ndarray): The gate matrix to apply.
            qubit (int): The index of the first qubit.
        """
        last_qubit_index = self.num_qubits - 1
        if qubit >= last_qubit_index:
            raise ValueError(
                "The double qubit can only be applied between neighboring qubits."
            )

        # Construct the full matrix using Kronecker products
        unitary = np.eye(1)  # Start with a scalar identity
        for q in range(self.num_qubits):
            if q == qubit:
                # Apply the SWAP gate matrix between the two neighboring qubits
                unitary = np.kron(unitary, gate)
            elif q == qubit + 1:
                # Skip the next qubit since gate already covered it
                continue
            else:
                # For all other qubits, apply the identity matrix
                unitary = np.kron(unitary, I)

        # Apply the constructed gate to the state vector
        self.state = np.dot(unitary, self.state)
