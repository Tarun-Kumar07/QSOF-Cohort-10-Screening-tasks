import numpy as np

from simulator.quantum_circuit import QuantumCircuit

# Hadamard gate
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

# X gate (Pauli-X)
X = np.array([[0, 1], [1, 0]])

I = np.eye(2)

""" 
 |control⟩ ----●---- 
               |
 |target⟩  ----X----
"""
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

# SWAP gate
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def zero_state(num_qubits: int) -> np.ndarray:
    """Returns state when all qubits are in state |0>"""
    state = np.zeros(2**num_qubits)
    state[0] = 1
    return state


def is_power_of_2(n) -> bool:
    """Check if n is a power of 2."""
    return (n & (n - 1)) == 0


class MatrixQuantumCircuit(QuantumCircuit):
    """
    A quantum circuit representation that uses matrices to simulate the evolution of
    a quantum state. Supports basic quantum gate operations such as H, X, CNOT, and SWAP.
    """

    def __init__(self, num_qubits: int, state: np.ndarray):
        """
        Initialize the circuit with a specified number of qubits and an initial state.

        Parameters:
            num_qubits (int): The number of qubits in the circuit.
            state (np.ndarray): The initial state vector.
        """
        self.num_qubits = num_qubits
        self.state = state

    @classmethod
    def initialize_num_qubits(cls, num_qubits: int) -> QuantumCircuit:
        """
        Create a circuit with the specified number of qubits, initialized to the zero state.

        Parameters:
            num_qubits (int): The number of qubits for the circuit.

        Returns:
            QuantumCircuit: A circuit initialized to the zero state.
        """
        if num_qubits < 1:
            raise ValueError("num_qubits must be greater than or equal to 1.")

        state = zero_state(num_qubits)
        return cls(num_qubits=num_qubits, state=state)

    @classmethod
    def initialize_state(cls, state: np.ndarray) -> QuantumCircuit:
        """
        Create a circuit initialized to a given state.

        Parameters:
            state (np.ndarray): The initial state vector, which must be normalized.

        Returns:
            QuantumCircuit: A circuit initialized with the provided state.
        """
        if not np.allclose(np.linalg.norm(state), 1):
            raise ValueError("state must be normalized")

        len_state = len(state)
        if len_state < 2 or not is_power_of_2(len_state):
            raise ValueError("len(state) must be a power of 2")

        num_qubits = int(np.log2(len_state))
        return cls(num_qubits=num_qubits, state=state)

    def get_state(self) -> np.ndarray:
        """
        Get the current state vector of the circuit.

        Returns:
            np.ndarray: The current state vector.
        """
        return self.state

    def h(self, qubit: int):
        self.__apply_single_qubit_gate(H, qubit)

    def x(self, qubit: int):
        self.__apply_single_qubit_gate(X, qubit)

    def __apply_single_qubit_gate(self, gate: np.ndarray, qubit: int):
        """
        Apply a single-qubit gate to a specified qubit.

        Parameters:
            gate (np.ndarray): The gate matrix to apply.
            qubit (int): The index of the qubit to apply the gate to.
        """
        unitary = np.eye(1)
        for q in range(self.num_qubits):
            q_gate = gate if q == qubit else I
            unitary = np.kron(unitary, q_gate)

        self.state = np.dot(unitary, self.state)

    def cnot(self, control_qubit: int, target_qubit: int):
        if control_qubit == target_qubit:
            raise ValueError("Control qubit cannot be the target qubit")

        # control qubit and target qubit are far away
        if control_qubit < target_qubit:
            # bring control qubit closer to target qubit using swap gates
            for q in range(control_qubit, target_qubit - 1):
                self.__swap_with_neighbours(q, q + 1)
            # control qubit is just above target qubit
            self.__apply_double_qubit_gate_with_next_qubit(CNOT, target_qubit - 1)
            for q in range(target_qubit - 1, control_qubit, -1):
                self.__swap_with_neighbours(q, q - 1)
        else:
            # bring target qubit closer to control qubit using swap gates
            # note there is an extra swap to take target qubit below control qubit
            for q in range(target_qubit, control_qubit):
                self.__swap_with_neighbours(q, q + 1)
            # control qubit is now just above provided value
            self.__apply_double_qubit_gate_with_next_qubit(CNOT, control_qubit - 1)
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
            raise ValueError("The SWAP gate can only be applied between neighboring qubits.")

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
            raise ValueError("The double qubit can only be applied between neighboring qubits.")

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
