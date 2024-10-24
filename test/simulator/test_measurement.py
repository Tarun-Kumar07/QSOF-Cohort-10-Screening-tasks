import numpy as np
import pytest

from simulator import (
    SimulatorType,
    create_quantum_circuit_from_qubits,
    expectation,
    sample,
)


class TestSampling:

    def test_sampling_with_uniform_superposition(self):
        """This is smoke test to check that sampling is working"""
        qc = create_quantum_circuit_from_qubits(SimulatorType.TENSOR, 2)
        qc.h(0)
        qc.h(1)

        np.random.seed(42)
        samples = sample(qc, 1000)

        expected = {"00": 269, "01": 234, "10": 254, "11": 243}
        assert expected == samples

    def test_sampling_with_10_state(
        self,
    ):
        """Test |00> => |10>
        |0⟩ --------- |0⟩

        |0⟩ ----X---- |1⟩
        """
        qc = create_quantum_circuit_from_qubits(SimulatorType.TENSOR, 2)

        qc.x(1)

        np.random.seed(42)
        samples = sample(qc, 1000)

        expected = {"10": 1000}
        assert expected == samples

    def test_sampling_with_negative_values(self):
        """Sampling with negative values must raise an exception"""
        qc = create_quantum_circuit_from_qubits(SimulatorType.MATRIX, 2)
        qc.h(0)
        qc.h(1)

        np.random.seed(42)

        with pytest.raises(ValueError, match="Sample count cannot be negative"):
            sample(qc, -1000)


class TestExpectation:

    def test_multi_qubit_pauli_word(self):
        pauli_word = {0: "X", 1: "Y"}
        qc = create_quantum_circuit_from_qubits(SimulatorType.TENSOR, 2)

        actual_exp = expectation(qc, pauli_word)
        assert 0 == actual_exp

    def test_multi_qubit_with_one_ideal(self):
        pauli_word = {0: "Z"}
        qc = create_quantum_circuit_from_qubits(SimulatorType.TENSOR, 2)

        actual_exp = expectation(qc, pauli_word)
        assert 1 == actual_exp

    def test_invalid_pauli(self):
        pauli_word = {0: "I"}
        qc = create_quantum_circuit_from_qubits(SimulatorType.TENSOR, 2)

        with pytest.raises(ValueError):
            expectation(qc, pauli_word)
