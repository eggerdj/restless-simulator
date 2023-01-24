import numpy as np
from qiskit.quantum_info import Operator as QiskitOperator


class Operator(QiskitOperator):
    """A qutrit compatible operator class.

    The :class:`Operator` class supports qutrit operators identified by unique labels. If the label
    does not identify a qutrit operator, then :meth:`from_label` will defer to the Qiskit
    implementation (i.e., :meth:`qiskit.quantum_info.Operator.from_label`).

    .. note::
        :class:`Operator` currently only supports single-qutrit labels for qutrit operator labels.

    The following labels are accepted, over-and-above those in
    :class:`qiskit.quantum_info.Operator`:

    - BN: The Nth Barg-matrix qutrit operator. See references for more details.

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
        BARG_X = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)
        BARG_W = np.exp(2j * np.pi / 3)
        BARG_Z = np.diag([1, BARG_W, BARG_W**2])
        BARG = {
            "B0": np.eye(3, dtype=complex),
            "B1": BARG_X,
            "B2": BARG_Z,
            "B3": BARG_X @ BARG_X,
            "B4": BARG_X @ BARG_Z,
            "B5": BARG_X @ BARG_X @ BARG_Z,
            "B6": BARG_X @ BARG_Z @ BARG_Z,
            "B7": BARG_X @ BARG_X @ BARG_Z @ BARG_Z,
            "B8": BARG_Z @ BARG_Z,
        }
        if label in BARG:
            return BARG[label]
        ##
        
        # Defer to QiskitOperator.from_label
        return super().from_label(label)
