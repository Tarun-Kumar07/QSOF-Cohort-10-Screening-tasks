# QOSF Cohort 10 Screening Task

This repository contains the submission for Task 1, which involves statevector simulation of quantum circuits. The problem statement is to simulate quantum circuits using matrix multiplication and tensor multiplication, and then compare the results. Below are the key questions addressed before diving into the coding:

1. How should qubits be represented?
2. How do the two simulators work, and why might one be better than the other?
3. How can we verify that the simulators are functioning correctly?
4. What experiments can be conducted to determine why one simulator is better?
5. Bonus questions...

The answers to these questions are explained in detail below, followed by an overview of the code design.

## Q1. How should qubits be represented?
- The state of qubits is represented as vectors, and there are two main conventions for ordering qubits:
    - **Little-endian**
      - The least significant qubit is considered the first qubit.
      - For example, \( \ket{6} = \ket{110} \).
    - **Big-endian**
      - The most significant qubit is considered the first qubit.
      - For example, \( \ket{6} = \ket{011} \).

- I chose to represent qubits using the little-endian convention because it aligns with how numbers are typically converted to binary strings.
- Additionally, Qiskit uses this same convention, making verification easier.

## Q2. How do the simulators work, and why might one be better than the other?




## Q3. How can we verify that the simulators are functioning correctly?




## Q4. What experiments can be conducted to determine why one simulator is better?




## Q5. Bonus Questions...




## Code Design

