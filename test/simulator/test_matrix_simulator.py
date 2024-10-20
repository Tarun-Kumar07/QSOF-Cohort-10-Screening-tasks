import numpy as np
import numpy.testing as npt
import pytest

from simulator import (
    create_quantum_circuit_from_qubits,
    create_quantum_circuit_from_state,
)
from simulator.quantum_circuit import SimulatorType

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
class TestQuantumCircuitInitialization:
    """
    Test initialization methods to create matrix quantum ciruits
    """

    @pytest.mark.parametrize("num_qubits", [-2, -1, 0])
    def test_initialize_num_qubits_raises_error_for_non_positive_values(
        self, simulator_type, num_qubits
    ):
        """Test that initializing with non-positive num_qubits raises a ValueError."""
        with pytest.raises(
            ValueError, match="num_qubits must be greater than or equal to 1."
        ):
            create_quantum_circuit_from_qubits(simulator_type, num_qubits)

    @pytest.mark.parametrize("num_qubits", [1, 2])
    def test_initialize_num_qubits_succeeds_for_positive_values(
        self, simulator_type, num_qubits
    ):
        """Test that initializing with positive num_qubits does not raise an exception."""
        try:
            create_quantum_circuit_from_qubits(simulator_type, num_qubits)
        except Exception as exc:
            pytest.fail(f"Unexpected exception raised: {exc}")

    @pytest.mark.parametrize(
        "denormalized_state",
        [
            np.array([1, 1]),
            np.array([0, 0]),
            np.array([0.5, 0.5]),
            np.array([1, 2, 3, 4]),
        ],
    )
    def test_initialize_state_raises_error_for_non_normalized_states(
        self, simulator_type, denormalized_state
    ):
        """Test that initializing with a non-normalized state raises a ValueError."""
        with pytest.raises(ValueError, match="state must be normalized"):
            create_quantum_circuit_from_state(simulator_type, denormalized_state)

    @pytest.mark.parametrize(
        "invalid_length_state", [np.array([1]), np.array([1, 1, 1]) / np.sqrt(3)]
    )
    def test_initialize_state_raises_error_for_non_power_of_2_length(
        self, simulator_type, invalid_length_state
    ):
        """
        Test that initializing with a state whose length is not a power of 2
        raises a ValueError.
        """
        with pytest.raises(ValueError, match=r"len\(state\) must be a power of 2"):
            create_quantum_circuit_from_state(simulator_type, invalid_length_state)

    @pytest.mark.parametrize(
        "valid_state",
        [
            np.array([1, -1]) / np.sqrt(2),
            np.array([0.5, np.sqrt(3) / 2]),
            np.array([1, -1, 1, 1]) / 2,
        ],
    )
    def test_initialize_state_succeeds_for_valid_states(
        self, simulator_type, valid_state
    ):
        """Test that initializing with a valid state does not raise an exception."""
        try:
            create_quantum_circuit_from_state(simulator_type, valid_state)
        except Exception as exc:
            pytest.fail(f"Unexpected exception raised: {exc}")


@pytest.mark.parametrize("simulator_type", SIMULATOR_TYPES)
class TestXGate:
    """Test Not gate"""

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
        """Test IX |00> = |01>"""
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        qc.x(1)

        state_01 = np.kron(ZERO_STATE, ONE_STATE)
        npt.assert_array_equal(qc.get_state(), state_01)

    def test_x_10(self, simulator_type):
        """Test IX |10> = |11>"""
        state_10 = np.kron(ONE_STATE, ZERO_STATE)
        qc = create_quantum_circuit_from_state(simulator_type, state_10)
        qc.x(1)

        state_11 = np.kron(ONE_STATE, ONE_STATE)
        npt.assert_array_equal(qc.get_state(), state_11)

    def test_multiple_x_gate(self, simulator_type):
        """Test XX |00> = |11>"""
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        qc.x(0)
        qc.x(1)

        state_11 = np.array([0, 0, 0, 1])
        npt.assert_array_equal(qc.get_state(), state_11)


@pytest.mark.parametrize("simulator_type", SIMULATOR_TYPES)
class TestHGate:
    """Test Hadamard gate"""

    def test_h_0(self, simulator_type):
        """Test h|0> = |+>"""
        qc = create_quantum_circuit_from_qubits(simulator_type, 1)
        qc.h(0)

        npt.assert_array_equal(qc.get_state(), PLUS_STATE)

    def test_h_1(self, simulator_type):
        """Test x|1> = |->"""
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
        """Test IH |00> = |0+>"""
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        qc.h(1)

        state_0plus = np.kron(ZERO_STATE, PLUS_STATE)
        npt.assert_array_equal(qc.get_state(), state_0plus)

    def test_h_10(self, simulator_type):
        """Test IH |10> = |1+>"""
        state_10 = np.kron(ONE_STATE, ZERO_STATE)
        qc = create_quantum_circuit_from_state(simulator_type, state_10)
        qc.h(1)

        state_1plus = np.kron(ONE_STATE, PLUS_STATE)
        npt.assert_array_equal(qc.get_state(), state_1plus)

    def test_multiple_h_gate(self, simulator_type):
        """Test HH |00> = |++>"""
        qc = create_quantum_circuit_from_qubits(simulator_type, 2)
        qc.h(0)
        qc.h(1)

        state_plus_plus = np.kron(PLUS_STATE, PLUS_STATE)
        npt.assert_array_equal(qc.get_state(), state_plus_plus)


@pytest.mark.parametrize("simulator_type", SIMULATOR_TYPES)
class TestCXGate:
    """Test Controlled not gate"""

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

    def test_cx_with_minimum_qubits(self, simulator_type):
        """Test
        |1⟩ ----●---- |1⟩
                |
        |0⟩ ----X---- |1⟩
        """
        state_10 = np.kron(ONE_STATE, ZERO_STATE)
        qc = create_quantum_circuit_from_state(simulator_type, state_10)
        qc.cnot(0, 1)

        state_11 = np.kron(ONE_STATE, ONE_STATE)
        npt.assert_array_equal(qc.get_state(), state_11)

    def test_reverse_cx_with_minimum_qubits(self, simulator_type):
        """Test
        |0⟩ ----X---- |1⟩
                |
        |1⟩ ----●---- |1⟩
        """
        state_01 = np.kron(ZERO_STATE, ONE_STATE)
        qc = create_quantum_circuit_from_state(simulator_type, state_01)
        qc.cnot(1, 0)

        state_11 = np.kron(ONE_STATE, ONE_STATE)
        npt.assert_array_equal(qc.get_state(), state_11)

    def test_cx_with_far_qubits(self, simulator_type):
        """Test
        |1⟩ ----●---- |1⟩
                |
        |0⟩ --------- |0⟩
                |
        |0⟩ ----X---- |1⟩
        """
        state_100 = kron([ONE_STATE, ZERO_STATE, ZERO_STATE])
        qc = create_quantum_circuit_from_state(simulator_type, state_100)
        qc.cnot(0, 2)

        state_101 = kron([ONE_STATE, ZERO_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_101)

    def test_reverse_cx_with_far_qubits(self, simulator_type):
        """Test
        |0⟩ ----X---- |1⟩
                |
        |0⟩ --------- |0⟩
                |
        |1⟩ ----●---- |1⟩
        """
        state_001 = kron([ZERO_STATE, ZERO_STATE, ONE_STATE])
        qc = create_quantum_circuit_from_state(simulator_type, state_001)
        qc.cnot(2, 0)

        state_101 = kron([ONE_STATE, ZERO_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_101)

    def test_cx_with_neighbouring_qubits_and_ideal_qubits_at_end(self, simulator_type):
        """Test
        |1⟩ ----●----- |1⟩
                |
        |0⟩ ----X----- |1⟩

        |1⟩ ---------- |1⟩

        |1⟩ ---------- |1⟩
        """
        state_1011 = kron([ONE_STATE, ZERO_STATE, ONE_STATE, ONE_STATE])
        qc = create_quantum_circuit_from_state(simulator_type, state_1011)

        qc.cnot(0, 1)

        state_1111 = kron([ONE_STATE] * 4)
        npt.assert_array_equal(qc.get_state(), state_1111)

    def test_cx_with_neighbouring_qubits_and_ideal_qubits_at_begining(
        self, simulator_type
    ):
        """Test
        |1⟩ ---------- |1⟩

        |1⟩ ---------- |1⟩

        |1⟩ ----●----- |1⟩
                |
        |1⟩ ----X----- |0⟩
        """
        state_1111 = kron([ONE_STATE] * 4)
        qc = create_quantum_circuit_from_state(simulator_type, state_1111)

        qc.cnot(2, 3)

        state_1110 = kron([ONE_STATE, ONE_STATE, ONE_STATE, ZERO_STATE])
        npt.assert_array_equal(qc.get_state(), state_1110)

    def test_cx_with_neighbouring_qubits_at_center(self, simulator_type):
        """Test
        |1⟩ ---------- |1⟩

        |1⟩ ----●----- |1⟩
                |
        |0⟩ ----X----- |1⟩

        |1⟩ ---------- |1⟩
        """
        state_1101 = kron([ONE_STATE, ONE_STATE, ZERO_STATE, ONE_STATE])
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

        state_0111 = kron([ZERO_STATE, ONE_STATE, ONE_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_0111)

    def test_reverse_cx_with_neighbouring_qubits_and_ideal_qubits_at_begining(
        self, simulator_type
    ):
        """Test
        |1⟩ ---------- |1⟩

        |1⟩ ---------- |1⟩

        |1⟩ ----X----- |0⟩
                |
        |1⟩ ----●----- |1⟩
        """
        state_1111 = kron([ONE_STATE] * 4)
        qc = create_quantum_circuit_from_state(simulator_type, state_1111)

        qc.cnot(3, 2)

        state_1101 = kron([ONE_STATE, ONE_STATE, ZERO_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_1101)

    def test_reverse_cx_with_neighbouring_qubits_at_center(self, simulator_type):
        """Test
        |1⟩ ---------- |1⟩

        |1⟩ ----X----- |0⟩
                |
        |1⟩ ----●----- |1⟩

        |1⟩ ---------- |1⟩
        """
        state_1111 = kron([ONE_STATE] * 4)
        qc = create_quantum_circuit_from_state(simulator_type, state_1111)

        qc.cnot(2, 1)

        state_1011 = kron([ONE_STATE, ZERO_STATE, ONE_STATE, ONE_STATE])
        npt.assert_array_equal(qc.get_state(), state_1011)

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
