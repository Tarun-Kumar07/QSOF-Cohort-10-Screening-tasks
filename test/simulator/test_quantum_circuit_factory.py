import numpy as np
import pytest

from simulator import (
    SimulatorType,
    create_quantum_circuit_from_qubits,
    create_quantum_circuit_from_state,
)

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
