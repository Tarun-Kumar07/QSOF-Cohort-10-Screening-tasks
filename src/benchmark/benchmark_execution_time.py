import ast
import csv
import time
from typing import Callable, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator

from simulator import SimulatorType, create_quantum_circuit_from_qubits

DATASET_FILE_PATH = "./data/dataset.csv"
REPORT_FILE_PATH = "./data/report.csv"
QISKIT_STATE_VECTOR_SIMULATOR = StatevectorSimulator()
ROUDNING_DECIMAL_PLACES = 4


def time_execution(func: Callable) -> Callable:
    """Decorator to time a function execution and return its result."""

    def wrapper(*args, **kwargs) -> Tuple[float, np.ndarray]:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time_seconds = end_time - start_time
        execution_time_microseconds = round(
            execution_time_seconds * (10**6), ROUDNING_DECIMAL_PLACES
        )

        return execution_time_microseconds, result

    return wrapper


@time_execution
def run_quantum_circuit(
    simulator_type: SimulatorType, num_qubits: int, circuit
) -> np.ndarray:
    qc = create_quantum_circuit_from_qubits(simulator_type, num_qubits)
    for gate, qubits in circuit:
        if gate == "H":
            qc.h(qubits[0])
        elif gate == "X":
            qc.x(qubits[0])
        elif gate == "CNOT":
            qc.cnot(qubits[0], qubits[1])

    return qc.get_state()


@time_execution
def run_qiskit_circuit(num_qubits: int, circuit) -> np.ndarray:
    qc = QuantumCircuit(num_qubits)
    for gate, qubits in circuit:
        if gate == "H":
            qc.h(qubits[0])
        elif gate == "X":
            qc.x(qubits[0])
        elif gate == "CNOT":
            qc.cx(qubits[0], qubits[1])

    # Qiskit orders qubits in little endian way, and the simulator has been designed in big endian fashion
    # So flipping the orders to convert qiskit circuit to big endian notation
    qc = qc.reverse_bits()

    result = QISKIT_STATE_VECTOR_SIMULATOR.run(qc).result()
    state_vector = np.asarray(result.get_statevector(qc))

    return state_vector


def compute_fidelity(state1, state2):
    inner_product = np.dot(state1.conjugate(), state2)
    fidelity = np.abs(inner_product) ** 2

    return float(round(fidelity, ROUDNING_DECIMAL_PLACES))


def read_dataset():
    dataset = []
    with open(DATASET_FILE_PATH, mode="r") as file:
        reader = csv.reader(file)

        # Skip the header
        next(reader)

        # Read the data
        for row in reader:
            name = row[0]
            num_qubits = int(row[1])
            depth = int(row[2])
            circuit = ast.literal_eval(row[3])  # Safely convert the name back to a list

            dataset.append((name, num_qubits, depth, circuit))

    return dataset


def benchmark_dataset(dataset):
    report = []
    for name, num_qubits, depth, circuit in dataset:
        print(f"Executing {name}")
        qiskit_exec_time, qiskit_sv = run_qiskit_circuit(num_qubits, circuit)
        matrix_sim_exec_time, matrix_sim_sv = run_quantum_circuit(
            SimulatorType.MATRIX, num_qubits, circuit
        )
        tensor_sim_exec_time, tensor_sim_sv = run_quantum_circuit(
            SimulatorType.TENSOR, num_qubits, circuit
        )

        matrix_sim_fidelity = compute_fidelity(matrix_sim_sv, qiskit_sv)
        tensor_sim_fidelity = compute_fidelity(tensor_sim_sv, qiskit_sv)

        report.append(
            [
                name,
                num_qubits,
                depth,
                matrix_sim_fidelity,
                matrix_sim_exec_time,
                tensor_sim_fidelity,
                tensor_sim_exec_time,
                qiskit_exec_time,
            ]
        )

    return report


def write_report(report):
    header = [
        "Name",
        "Num Qubits",
        "Depth",
        "Matrix Sim Fidelity",
        "Matrix Sim Exec Time (ms)",
        "Tensor Sim Fidelity",
        "Tensor Sim Exec Time (ms)",
        "Qiskit Exec Time (ms)",
    ]

    with open(REPORT_FILE_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header
        for row in report:
            writer.writerow([str(d) for d in row])


def main():
    dataset = read_dataset()
    report = benchmark_dataset(dataset)
    write_report(report)


if __name__ == "__main__":
    main()
