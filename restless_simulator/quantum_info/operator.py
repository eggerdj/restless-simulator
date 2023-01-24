# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Operator class compatible with qutrit operator labels."""
import numpy as np
from qiskit.quantum_info import Operator as QiskitOperator


# pylint: disable=too-many-ancestors
class Operator(QiskitOperator):
    """A qutrit compatible operator class.

    The :class:`Operator` class supports qutrit operators identified by unique labels. If the label
    does not identify a qutrit operator, then :meth:`from_label` will defer to the Qiskit
    implementation (i.e., :meth:`qiskit.quantum_info.Operator.from_label`).

    .. note::
        :class:`Operator` currently only supports single-qutrit labels for qutrit operator labels.

    The following labels are accepted, over-and-above those in
    :class:`qiskit.quantum_info.Operator`:

    - BN: The Nth Barg-matrix qutrit operator (:math:`N\\in\\{0,1,\\ldots,8\\}`). See references for
      more details.

    .. warning::

        This class supports operators of arbitrary sizes and not just qutrit operators. Make sure
        you are working with an appropriately sized operator for your code.

    References:
        [1] A. Barg, ‘A low-rate bound on the reliability of a quantum discrete memoryless channel’,
        IEEE Transactions on Information Theory, vol. 48, no. 12, pp. 3096–3100, Dec. 2002, doi:
        10.1109/TIT.2002.805080.
    """

    @classmethod
    def from_label(cls, label):
        """Returns an operator for the given label.

        Args:
            label: the operator label.

        Returns:
            QiskitOperator: The operator for the label.
        """
        ## Barg single-qutrit matrices
        barg_x = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)
        barg_w = np.exp(2j * np.pi / 3)
        barg_z = np.diag([1, barg_w, barg_w**2])
        barg = {
            "B0": np.eye(3, dtype=complex),
            "B1": barg_z,
            "B2": barg_z @ barg_z,
            "B3": barg_x,
            "B4": barg_x @ barg_z,
            "B5": barg_x @ barg_z @ barg_z,
            "B6": barg_x @ barg_x,
            "B7": barg_x @ barg_x @ barg_z,
            "B8": barg_x @ barg_x @ barg_z @ barg_z,
        }
        if label in barg:
            return barg[label]
        ##

        # Defer to QiskitOperator.from_label
        return super().from_label(label)
