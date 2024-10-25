"""
This script is used to benchmark the number of qubits a simulator can simulate.
"""

from benchmark.benchmark_execution_time import run_quantum_circuit
from benchmark.dataset_generator import generate_random_circuit
from simulator import SimulatorType

DEPTH = 10
MAX_QUBITS = 30


def print_execution_time(simulator_type: SimulatorType) -> int:
    for qubits in range(2, MAX_QUBITS):
        circuit = generate_random_circuit(qubits, DEPTH)
        execution_time, _ = run_quantum_circuit(simulator_type, qubits, circuit)
        print(
            f" Simulator of type {simulator_type} took {execution_time * (10 ** -6)} for {qubits} qubits"
        )


if __name__ == "__main__":
    # print_execution_time(SimulatorType.MATRIX)
    print_execution_time(SimulatorType.TENSOR)
