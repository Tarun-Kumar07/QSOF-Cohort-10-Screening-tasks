import numpy as np
import numpy.testing as npt
import pytest

from simulator import (
    SimulatorType,
    create_quantum_circuit_from_qubits,
    create_quantum_circuit_from_state,
)

ZERO_STATE = np.array([1, 0])

ONE_STATE = np.array([0, 1])

MINUS_STATE = np.array([1, -1]) / np.sqrt(2)

PLUS_STATE = np.array([1, 1]) / np.sqrt(2)


def kron(arrays: list[np.ndarray]) -> np.ndarray:
    """Compute the Kronecker product of a list of NumPy arrays."""
    if not arrays:
        raise ValueError("Input list must contain at least one array.")

    # Start with the first array
    result = arrays[0]

    # Iterate through the rest of the arrays and compute the Kronecker product
    for array in arrays[1:]:
        result = np.kron(result, array)

    return result


SIMULATOR_TYPES = [st for st in SimulatorType]


@pytest.mark.parametrize("simulator_type", SIMULATOR_TYPES)
class TestXGate:
    """Test Not gate"""

    @pytest.mark.parametrize("qubit", [-1, 3])
    def test_x_with_invalid_qubits_raises_error(self, simulator_type, qubit):
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        with pytest.raises(ValueError, match=r"qubit must be in \[0, 1\]"):
            qc.x(qubit)

    def test_x_0(self, simulator_type):
        """Test x|0> = |1>"""
        qc = create_quantum_circuit_from_qubits(simulator_type, 1)
        qc.x(0)

        npt.assert_array_equal(qc.get_state(), ONE_STATE)

    def test_x_1(self, simulator_type):
        """Test x|1> = |0>"""
        qc = create_quantum_circuit_from_state(simulator_type, ONE_STATE)
        qc.x(0)

        zero_state = np.array([1, 0])
        npt.assert_array_equal(qc.get_state(), zero_state)

    def test_x_plus(self, simulator_type):
        """Test x|+> = |+>"""
        qc = create_quantum_circuit_from_state(simulator_type, PLUS_STATE)
        qc.x(0)

        npt.assert_array_equal(qc.get_state(), PLUS_STATE)

    def test_x_minus(self, simulator_type):
        """Test x|-> = |->"""
        qc = create_quantum_circuit_from_state(simulator_type, MINUS_STATE)
        qc.x(0)

        expected_state = (-1) * MINUS_STATE  # Global phase of -1
        npt.assert_array_equal(qc.get_state(), expected_state)

    def test_x_00(self, simulator_type):
        """Test |00> => |10>
        |0⟩ --------- |0⟩

        |0⟩ ----X---- |1⟩
        """
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        qc.x(1)

        state_10 = np.kron(ONE_STATE, ZERO_STATE)
        npt.assert_array_equal(qc.get_state(), state_10)

    def test_x_10(self, simulator_type):
        """Test |01> => |11>
        |1⟩ --------- |1⟩

        |0⟩ ----X---- |1⟩
        """
        state_01 = np.kron(ZERO_STATE, ONE_STATE)
        qc = create_quantum_circuit_from_state(simulator_type, state_01)
        qc.x(1)

        state_11 = np.kron(ONE_STATE, ONE_STATE)
        npt.assert_array_equal(qc.get_state(), state_11)

    def test_multiple_x_gate(self, simulator_type):
        """Test |00> => |11>
        |0⟩ --------- |1⟩

        |0⟩ ----X---- |1⟩
        """
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        qc.x(0)
        qc.x(1)

        state_11 = np.array([0, 0, 0, 1])
        npt.assert_array_equal(qc.get_state(), state_11)


@pytest.mark.parametrize("simulator_type", SIMULATOR_TYPES)
class TestHGate:
    """Test Hadamard gate"""

    @pytest.mark.parametrize("qubit", [-1, 3])
    def test_h_with_invalid_qubits_raises_error(self, simulator_type, qubit):
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        with pytest.raises(ValueError, match=r"qubit must be in \[0, 1\]"):
            qc.h(qubit)

    def test_h_0(self, simulator_type):
        """Test H|0> = |+>"""
        qc = create_quantum_circuit_from_qubits(simulator_type, 1)
        qc.h(0)

        npt.assert_array_equal(qc.get_state(), PLUS_STATE)

    def test_h_1(self, simulator_type):
        """Test H|1> = |->"""
        qc = create_quantum_circuit_from_state(simulator_type, ONE_STATE)
        qc.h(0)

        npt.assert_array_equal(qc.get_state(), MINUS_STATE)

    def test_h_plus(self, simulator_type):
        """Test H|+> = |0>"""
        qc = create_quantum_circuit_from_state(simulator_type, PLUS_STATE)
        qc.h(0)

        npt.assert_array_almost_equal(qc.get_state(), ZERO_STATE)

    def test_h_minus(self, simulator_type):
        """Test H|-> = |1>"""
        qc = create_quantum_circuit_from_state(simulator_type, MINUS_STATE)
        qc.h(0)

        npt.assert_array_almost_equal(qc.get_state(), ONE_STATE)

    def test_h_00(self, simulator_type):
        """Test |00> => |+0>
        |0⟩ --------- |0⟩

        |0⟩ ----H---- |+⟩
        """
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        qc.h(1)

        state_plus0 = np.kron(PLUS_STATE, ZERO_STATE)
        npt.assert_array_equal(qc.get_state(), state_plus0)

    def test_h_10(self, simulator_type):
        """Test |01> => |+1>
        |1⟩ --------- |1⟩

        |0⟩ ----H---- |+⟩
        """
        state_01 = np.kron(ZERO_STATE, ONE_STATE)
        qc = create_quantum_circuit_from_state(simulator_type, state_01)
        qc.h(1)

        state_plus1 = np.kron(PLUS_STATE, ONE_STATE)
        npt.assert_array_equal(qc.get_state(), state_plus1)

    def test_multiple_h_gate(self, simulator_type):
        """Test |00> => |++>
        |0⟩ ----H---- |+⟩

        |0⟩ ----H---- |+⟩
        """
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        qc.h(0)
        qc.h(1)

        state_plus_plus = np.kron(PLUS_STATE, PLUS_STATE)
        npt.assert_array_equal(qc.get_state(), state_plus_plus)


@pytest.mark.parametrize("simulator_type", SIMULATOR_TYPES)
class TestCnotGate:
    """Test Controlled not gate"""

    @pytest.mark.parametrize("target", [-2, 2])
    @pytest.mark.parametrize("control", [-1, 3])
    def test_cnot_with_invalid_qubits_raises_error(
        self, simulator_type, control, target
    ):
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        with pytest.raises(ValueError, match=r"qubit must be in \[0, 1\]"):
            qc.cnot(control, target)

    def test_exception_when_control_and_target_qubit_same(self, simulator_type):
        """Test exception is thrown when control and target qubit are same"""
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        with pytest.raises(
            ValueError, match="Control qubit cannot be the target qubit"
        ):
            qc.cnot(0, 0)

    def test_control_qubit_disabled(self, simulator_type):
        """Test
        |0⟩ ----●---- |0⟩
                |
        |0⟩ ----X---- |0⟩
        """
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)

        qc.cnot(0, 1)

        state_00 = np.kron(ZERO_STATE, ZERO_STATE)
        npt.assert_array_equal(qc.get_state(), state_00)

    def test_cx_with_11(self, simulator_type):
        """Test |11> => |01>
        |1⟩ ----●---- |1⟩
                |
        |1⟩ ----X---- |0⟩
        """
        state_11 = np.kron(ONE_STATE, ONE_STATE)
        qc = create_quantum_circuit_from_state(simulator_type, state_11)
        qc.cnot(0, 1)

        state_01 = np.kron(ZERO_STATE, ONE_STATE)
        npt.assert_array_equal(qc.get_state(), state_01)

    def test_cx_with_01(self, simulator_type):
        """Test |01> => |11>
        |1⟩ ----●---- |1⟩
                |
        |0⟩ ----X---- |1⟩
        """
        state_01 = np.kron(ZERO_STATE, ONE_STATE)
        qc = create_quantum_circuit_from_state(simulator_type, state_01)
        qc.cnot(0, 1)

        state_11 = np.kron(ONE_STATE, ONE_STATE)
        npt.assert_array_equal(qc.get_state(), state_11)

    def test_reverse_cx_with_minimum_qubits(self, simulator_type):
        """Test |10> => |11>
        |0⟩ ----X---- |1⟩
                |
        |1⟩ ----●---- |1⟩
        """
        state_10 = np.kron(ONE_STATE, ZERO_STATE)
        qc = create_quantum_circuit_from_state(simulator_type, state_10)
        qc.cnot(1, 0)

        state_11 = np.kron(ONE_STATE, ONE_STATE)
        npt.assert_array_equal(qc.get_state(), state_11)

    def test_cx_with_far_qubits(self, simulator_type):
        """Test |001> => |101>
        |1⟩ ----●---- |1⟩
                |
        |0⟩ --------- |0⟩
                |
        |0⟩ ----X---- |1⟩
        """
        state_001 = kron([ZERO_STATE, ZERO_STATE, ONE_STATE])
        qc = create_quantum_circuit_from_state(simulator_type, state_001)
        qc.cnot(0, 2)

        state_101 = kron([ONE_STATE, ZERO_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_101)

    def test_reverse_cx_with_far_qubits(self, simulator_type):
        """Test |100> => |101>
        |0⟩ ----X---- |1⟩
                |
        |0⟩ --------- |0⟩
                |
        |1⟩ ----●---- |1⟩
        """
        state_100 = kron([ONE_STATE, ZERO_STATE, ZERO_STATE])
        qc = create_quantum_circuit_from_state(simulator_type, state_100)
        qc.cnot(2, 0)

        state_101 = kron([ONE_STATE, ZERO_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_101)

    def test_cx_with_neighbouring_qubits_and_ideal_qubits_at_end(self, simulator_type):
        """Test |1101> => |1111>
        |1⟩ ----●----- |1⟩
                |
        |0⟩ ----X----- |1⟩

        |1⟩ ---------- |1⟩

        |1⟩ ---------- |1⟩
        """
        state_1101 = kron([ONE_STATE, ONE_STATE, ZERO_STATE, ONE_STATE])
        qc = create_quantum_circuit_from_state(simulator_type, state_1101)

        qc.cnot(0, 1)

        state_1111 = kron([ONE_STATE] * 4)
        npt.assert_array_equal(qc.get_state(), state_1111)

    def test_cx_with_neighbouring_qubits_and_ideal_qubits_at_begining(
        self, simulator_type
    ):
        """Test |111> => |0111>
        |1⟩ ---------- |1⟩

        |1⟩ ---------- |1⟩

        |1⟩ ----●----- |1⟩
                |
        |1⟩ ----X----- |0⟩
        """
        state_1111 = kron([ONE_STATE] * 4)
        qc = create_quantum_circuit_from_state(simulator_type, state_1111)

        qc.cnot(2, 3)

        state_0111 = kron([ZERO_STATE, ONE_STATE, ONE_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_0111)

    def test_cx_with_neighbouring_qubits_at_center(self, simulator_type):
        """Test |1011> => |1111>
        |1⟩ ---------- |1⟩

        |1⟩ ----●----- |1⟩
                |
        |0⟩ ----X----- |1⟩

        |1⟩ ---------- |1⟩
        """
        state_1101 = kron([ONE_STATE, ZERO_STATE, ONE_STATE, ONE_STATE])
        qc = create_quantum_circuit_from_state(simulator_type, state_1101)

        qc.cnot(1, 2)

        state_1111 = kron([ONE_STATE] * 4)
        npt.assert_array_equal(qc.get_state(), state_1111)

    def test_reverse_cx_with_neighbouring_qubits_and_ideal_qubits_at_end(
        self, simulator_type
    ):
        """Test
        |1⟩ ----X----- |0⟩
                |
        |1⟩ ----●----- |1⟩

        |1⟩ ---------- |1⟩

        |1⟩ ---------- |1⟩
        """
        state_1111 = kron([ONE_STATE] * 4)
        qc = create_quantum_circuit_from_state(simulator_type, state_1111)

        qc.cnot(1, 0)

        state_1110 = kron([ONE_STATE, ONE_STATE, ONE_STATE, ZERO_STATE])
        npt.assert_array_equal(qc.get_state(), state_1110)

    def test_reverse_cx_with_neighbouring_qubits_and_ideal_qubits_at_begining(
        self, simulator_type
    ):
        """Test |1111> => |1011>
        |1⟩ ---------- |1⟩

        |1⟩ ---------- |1⟩

        |1⟩ ----X----- |0⟩
                |
        |1⟩ ----●----- |1⟩
        """
        state_1111 = kron([ONE_STATE] * 4)
        qc = create_quantum_circuit_from_state(simulator_type, state_1111)

        qc.cnot(3, 2)

        state_1011 = kron([ONE_STATE, ZERO_STATE, ONE_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_1011)

    def test_reverse_cx_with_neighbouring_qubits_at_center(self, simulator_type):
        """Test |1111> => |1101>
        |1⟩ ---------- |1⟩

        |1⟩ ----X----- |0⟩
                |
        |1⟩ ----●----- |1⟩

        |1⟩ ---------- |1⟩
        """
        state_1111 = kron([ONE_STATE] * 4)
        qc = create_quantum_circuit_from_state(simulator_type, state_1111)

        qc.cnot(2, 1)

        state_1101 = kron([ONE_STATE, ONE_STATE, ZERO_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_1101)

    def test_multiple_cnot_gates(self, simulator_type):
        """Test : Create Swap gate using CNOT
        |1⟩ --X--●--X-- |0⟩
              |  |  |
        |0⟩ --●--X--●-- |1⟩
        """
        state_10 = kron([ONE_STATE, ZERO_STATE])
        qc = create_quantum_circuit_from_state(simulator_type, state_10)

        qc.cnot(1, 0)
        qc.cnot(0, 1)
        qc.cnot(1, 0)

        state_01 = kron([ZERO_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_01)


@pytest.mark.parametrize("simulator_type", SIMULATOR_TYPES)
def test_generate_GHZ_state(simulator_type):
    """Test : Generate GHZ state"""
    qc = create_quantum_circuit_from_qubits(simulator_type, 3)
    qc.h(0)
    qc.cnot(0, 1)
    qc.cnot(1, 2)

    ghz_state = np.zeros(8)
    ghz_state[0] = 1 / np.sqrt(2)
    ghz_state[-1] = 1 / np.sqrt(2)

    npt.assert_array_equal(qc.get_state(), ghz_state)
