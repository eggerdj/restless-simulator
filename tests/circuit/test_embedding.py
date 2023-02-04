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

from qiskit.circuit.library import RZZGate

from restless_simulator.circuit import QutritUnitaryGate
from restless_simulator.circuit.qutrit_unitary_gate import convert_basis_index


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
