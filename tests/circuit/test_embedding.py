# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test circuit embedding methods."""

from unittest import TestCase
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZZGate, CPhaseGate
from qiskit import quantum_info as qi

from restless_simulator.circuit import QutritUnitaryGate
from restless_simulator.circuit.qutrit_unitary_gate import convert_basis_index
from restless_simulator.quantum_info.converters import (
    qudit_circuit_to_super_op,
    circuit_to_qudit_circuit,
)


class TestEmbedding(TestCase):
    """Test the embedding of qubit-based unitaries into larger ones."""

    def test_convert_to_qutrit(self):
        """Test the index conversion for qutrits."""

        qubit_indices = [0, 1, 2, 3]  # correspond to 00, 01, 10, 11, respectively.

        # The SU(3**2) states are 00, 01, 02, 10, 11, 12, 20, 21, 22.
        qutrit_indices = [0, 1, 3, 4]

        for j, index in enumerate(qubit_indices):
            self.assertEqual(convert_basis_index(index, basis=3), qutrit_indices[j])

    def test_two_qutrit_gate(self):
        """Test the embedding of an rzz gate."""
        qubit_rzz = RZZGate(np.pi)
        qutrit_rzz = QutritUnitaryGate.from_qubit_gate(qubit_rzz)

        qutrit_mat = qutrit_rzz.to_matrix()
        ideal_diag = [-1.0j, 1.0j, 1.0, 1.0j, -1.0j, 1.0, 1.0, 1.0, 1.0]

        self.assertTrue(np.allclose(np.diag(qutrit_mat), ideal_diag))

    def test_correct_evolution(self):
        """A test of a specific instance."""

        qt_ry = QutritUnitaryGate.from_qubit_gate(RYGate(0.5 * np.pi, label="ry"))
        qt_rx = QutritUnitaryGate.from_qubit_gate(RXGate(np.pi, label="rx"))
        qt_rzz = QutritUnitaryGate.from_qubit_gate(CPhaseGate(np.pi, label="cp"))

        circuit = QuantumCircuit(2)
        circuit.append(qt_ry, (0,))
        circuit.append(qt_rx, (1,))
        circuit.append(qt_rzz, (0, 1))

        sop = qudit_circuit_to_super_op(circuit)

        # Create |0><0|
        data = np.zeros((9, 9))
        data[0, 0] = 1
        rho_in = qi.DensityMatrix(data)

        # Evolve with sop to get |10> - |11>
        rho_out = rho_in.evolve(sop).data

        sub_rho_out = np.array(
            [
                [rho_out[3, 3], rho_out[3, 4]],
                [rho_out[4, 3], rho_out[4, 4]],
            ]
        )

        expected = np.array(
            [
                [0.5, -0.5],
                [-0.5, 0.5],
            ]
        )

        self.assertTrue(np.allclose(sub_rho_out, expected))

    def test_conversion(self):
        """Test that the circuit to qudit circuit conversion gives the same as a manual build."""

        # Automatic conversion
        circuit1 = QuantumCircuit(2)
        circuit1.ry(0.5 * np.pi, 0)
        circuit1.rx(np.pi, 1)
        circuit1.cp(np.pi, 0, 1)
        circuit1 = circuit_to_qudit_circuit(circuit1)

        # Manual construction
        qt_ry = QutritUnitaryGate.from_qubit_gate(RYGate(0.5 * np.pi, label="ry"))
        qt_rx = QutritUnitaryGate.from_qubit_gate(RXGate(np.pi, label="rx"))
        qt_rzz = QutritUnitaryGate.from_qubit_gate(CPhaseGate(np.pi, label="cp"))

        circuit2 = QuantumCircuit(2)
        circuit2.append(qt_ry, (0,))
        circuit2.append(qt_rx, (1,))
        circuit2.append(qt_rzz, (0, 1))

        self.assertTrue(circuit1 == circuit2)
