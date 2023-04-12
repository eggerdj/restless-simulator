# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qutrit based circuit to SuperOp conversion."""

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RXGate, RZZGate

from unittest import TestCase

from restless_simulator.circuit import QutritUnitaryGate
from restless_simulator.quantum_info import qudit_circuit_to_super_op


class TestCircuitToSuperOp(TestCase):
    """Test conversion from circuit to SuperOp instances."""

    def test_multiple_wires(self):
        """Test conversion for a circuit with multiple wires."""
        qt_rx = QutritUnitaryGate.from_qubit_gate(RXGate(0.5 * np.pi, label="rx"))

        circuit = QuantumCircuit(3)
        circuit.append(qt_rx, (0,))
        circuit.append(qt_rx, (1,))

        super_op = qudit_circuit_to_super_op(circuit)

        self.assertEqual(super_op.dim, (27, 27))
        self.assertEqual(super_op.output_dims(), (3, 3, 3))

    def test_convert_with_two_qubit_gate(self):
        """Test the conversion with a circuit that has a two-qubit gate."""

        qt_rx = QutritUnitaryGate.from_qubit_gate(RXGate(0.5 * np.pi, label="rx"))
        qt_rzz = QutritUnitaryGate.from_qubit_gate(RZZGate(0.5 * np.pi, label="rzz"))

        circuit = QuantumCircuit(3)
        circuit.append(qt_rx, (0,))
        circuit.append(qt_rzz, (0, 2))

        super_op = qudit_circuit_to_super_op(circuit)

        self.assertEqual(super_op.dim, (27, 27))
        self.assertEqual(super_op.output_dims(), (3, 3, 3))
