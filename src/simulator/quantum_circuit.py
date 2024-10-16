from abc import ABC, abstractmethod


class QuantumCircuit(ABC):
    """
    Defines abstract class with abstract methods on operations on QuantumCircuit
    """

    @abstractmethod
    def h(self, qubit: int):
        """
        Apply the Hadamard gate to a specified qubit.

        Parameters:
            qubit (int): The index of the qubit to apply the Hadamard gate to.
        """

    @abstractmethod
    def x(self, qubit: int):
        """
        Apply the Pauli-X gate (NOT gate) to a specified qubit.

        Parameters:
            qubit (int): The index of the qubit to apply the Pauli-X gate to.
        """

    @abstractmethod
    def cnot(self, control_qubit: int, target_qubit: int):
        """
        Apply a controlled-NOT (CNOT) gate between two qubits.

        Parameters:
            control_qubit (int): The control qubit index.
            target_qubit (int): The target qubit index.
        """
