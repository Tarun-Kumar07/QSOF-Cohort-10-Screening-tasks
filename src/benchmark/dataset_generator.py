"""
This script generates dataset consisting of quantum circuits which will be used to benchmark the simulators
"""

import csv
import itertools
from typing import List, Tuple

import numpy as np

FILE_PATH = "./data/dataset.csv"

np.random.seed(42)


def generate_random_circuit(num_qubits: int, depth: int) -> List[Tuple[str, List[int]]]:
    """
    Generates a random quantum circuit with a specified number of qubits and depth.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        depth (int): The number of layers in the circuit.

    Returns:
        List[Tuple[str, List[int]]]: A list of tuples representing the circuit.
                                     Each tuple contains a gate name (str) and the qubits it acts on (List[int]).
    """
    gates = ["H", "X", "CNOT"]  # Allowed gate types
    circuit = []

    for _ in range(depth):
        gate = str(np.random.choice(gates))
        if gate == "CNOT":
            # For CNOT, randomly select two different qubits (control and target)
            control = np.random.randint(0, num_qubits)
            target = np.random.randint(0, num_qubits)
            # Ensure control and target are not the same
            while target == control:
                target = np.random.randint(0, num_qubits)
            circuit.append((gate, [control, target]))
        else:
            # For H or X, randomly select a single qubit
            qubit = np.random.randint(0, num_qubits)
            circuit.append((gate, [qubit]))

    return circuit


def create_dataset():
    qubits = [2, 4, 6, 8]
    depth = [2, 4, 8]
    num_circuits = 5
    data = []
    for num_qubits, depth in itertools.product(qubits, depth):
        for i in range(num_circuits):
            name = f"q_{num_qubits}_d_{depth}__{i}"
            circuit = generate_random_circuit(num_qubits, depth)
            data.append((name, num_qubits, depth, circuit))
    return data


def write_dataset(data):
    with open(FILE_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Num_qubits", "Depth", "Circuit"])

        # Write the data
        for count, (name, num_qubits, depth, circuit) in enumerate(data):
            print(f"{count} => Writing circuit {name}")
            writer.writerow([name, num_qubits, depth, circuit])


def main():
    data = create_dataset()
    write_dataset(data)


if __name__ == "__main__":
    main()
