# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test execution of simple circuits."""

from unittest import TestCase

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError

from restless_simulator.simulator import QutritRestlessSimulator


class TestQubitCircuitExecution(TestCase):
    """Test the execution of simple circuits."""

    def setUp(self):
        """Setup for tests."""
        self._backend = QutritRestlessSimulator()

    def test_single_qubit_x(self):
        """Test single-qubit x gate."""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        result = self._backend.run(circuit, shots=10).result()

        self.assertEqual(result.get_counts(0), {"0": 5, "1": 5})

    def test_single_qubit_id(self):
        """Test single-qubit x gate."""
        circuit = QuantumCircuit(1)

        result = self._backend.run(circuit, shots=10).result()

        self.assertEqual(result.get_counts(0), {"0": 10})

    def test_single_qubit_two_circuits(self):
        """Test two circuits, one with an id and the other with an x gate."""
        circuit1 = QuantumCircuit(1)
        circuit2 = QuantumCircuit(1)
        circuit2.x(0)

        result = self._backend.run([circuit1, circuit2], shots=10).result()

        self.assertEqual(result.get_counts(0), {"0": 5, "1": 5})
        self.assertEqual(result.get_counts(1), {"0": 5, "1": 5})

    def test_raise_on_incompatible_circuits(self):
        """Test that circuits with a different number of wires raises."""
        with self.assertRaises(QiskitError):
            self._backend.run([QuantumCircuit(1), QuantumCircuit(2)])

    def test_two_qubits(self):
        """Test the execution of circuits with two qubits."""
        circuit = QuantumCircuit(2)
        circuit.x(1)

        result = self._backend.run(circuit, shots=10).result()

        self.assertEqual(result.get_counts(0), {"00": 5, "10": 5})

    def test_two_qubit_gate(self):
        """Test a circuit with a two-qubit gate.

        Note: the circuit is chosen to be deterministic on purpose. Anything
        that involves entanglement is tricky to Test. Here, the X followed by
        a swap will make the "1"'s oscillate between the two qubits the pattern
        of measurement outcomes is 10 -> 11 -> 01 -> 00 -> repeat.
        """
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.swap(0, 1)

        result = self._backend.run(circuit, shots=10).result()
        expected = ["10", "11", "01", "00", "10", "11", "01", "00", "10", "11"]

        self.assertEqual(result.get_memory(), expected)
