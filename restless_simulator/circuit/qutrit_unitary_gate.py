# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Qutrit unitary gate.

This code is a modified version of the Qiskit implementation of a qubit unitary gate.
"""
from typing import Optional, Union

import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.gate import Gate
from qiskit.exceptions import QiskitError
from qiskit.extensions.exceptions import ExtensionError
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.predicates import is_unitary_matrix, matrix_equal


class QutritUnitaryGate(Gate):
    """Quantum gates specified by a unitary matrix for qutrits."""

    ATOL_DEFAULT = 1e-8
    """Absolute tolerance for unitary check."""
    RTOL_DEFAULT = 1e-7
    """Relative tolerance for unitary check"""

    def __init__(
        self,
        data: Union[np.ndarray, Gate, Operator],
        label: Optional[str] = None,
        atol: float = ATOL_DEFAULT,
        rtol: float = RTOL_DEFAULT,
    ):
        """Create a gate from a numeric unitary matrix.

        Args:
            data (matrix or Operator): unitary operator.
            label (str): unitary name for backend [Default: None].
            atol: absolute tolerance for unitary check. Defaults to :attr:`ATOL_DEFAULT`.
            rtol: relative tolerance for unitary check. Defaults to :attr:`RTOL_DEFAULT`.

        Raises:
            ExtensionError: if input data is not an N-qubit unitary operator.
        """
        self._atol = atol
        self._rtol = rtol

        if hasattr(data, "to_matrix"):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            data = data.to_matrix()
        elif hasattr(data, "to_operator"):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            data = data.to_operator().data

        # Convert to numpy array in case not already an array
        data = np.array(data, dtype=complex)

        # Check input is unitary
        if not is_unitary_matrix(data, rtol=self._rtol, atol=self._atol):
            raise ExtensionError("Input matrix is not unitary.")

        # Check input is N-qubit matrix
        input_dim, output_dim = data.shape
        num_qutrits = int(np.log(input_dim) / np.log(3))
        if input_dim != output_dim or 3**num_qutrits != input_dim:
            raise ExtensionError("Input matrix is not an N-qutrit operator.")

        self._qasm_name = None
        self._qasm_definition = None
        super().__init__("unitary", num_qutrits, [data], label=label)

    def __eq__(self, other):
        if not isinstance(other, QutritUnitaryGate):
            return False
        if self.label != other.label:
            return False
        return matrix_equal(self.params[0], other.params[0], ignore_phase=True)

    def __array__(self, dtype=None):
        """Return matrix for the unitary."""
        # pylint: disable=unused-argument
        return self.params[0]

    def inverse(self):
        """Return the adjoint of the unitary."""
        return self.adjoint()

    def conjugate(self):
        """Return the conjugate of the unitary."""
        return QutritUnitaryGate(np.conj(self.to_matrix()))

    def adjoint(self):
        """Return the adjoint of the unitary."""
        return self.transpose().conjugate()

    def transpose(self):
        """Return the transpose of the unitary."""
        return QutritUnitaryGate(np.transpose(self.to_matrix()))

    def validate_parameter(self, parameter):
        """Unitary gate parameter has to be an ndarray."""
        if isinstance(parameter, np.ndarray):
            return parameter
        raise CircuitError(f"invalid param type {type(parameter)} in gate {self.name}")

    def as_operator(self) -> Operator:
        """Returns the gate as an :class:`Operator`.

        This function is useful when simulating the circuit.

        Returns:
            Operator: the operator representation of this gate.
        """
        return Operator(np.asarray(self))

    @classmethod
    def from_qubit_gate(
        cls, qubit_gate: Gate, label: Optional[str] = None
    ) -> "QutritUnitaryGate":
        r"""Create a qutrit gate from the provided qubit gate.

        This function creates a qutrit unitary gate for a unitary :math:`U`. For example, a
        single-qubit unitary is embedded in the following structure

        .. math::
            \begin{bmatrix}
            a_{0,0} & a_{0,1} & 0 \\
            a_{1,0} & a_{1,1} & 0 \\
            0       & 0       & 1 \\
            \end{bmatrix}

        where :math:`a_{i,j}` is the unitary element of the input gate for the ith row and jth
        column. If the input gate is not unitary, an error will be thrown as the resulting qutrit
        gate will be non-unitary.

        Args:
            qubit_gate: The qubit gate to convert into a qutrit gate.
            label: Optional label for the new qutrit gate. Defaults to None.

        Raises:
            QiskitError: if the input gate is not for a qubit.
            QiskitError: if the resulting qutrit gate is non-unitary.

        Returns:
            QutritUnitaryGate: a qutrit gate equivalent to the input qubit gate.
        """
        qubit_unitary = qubit_gate.to_matrix()

        num_qubits = float(np.log2(qubit_unitary.shape[0]))

        if not num_qubits.is_integer() or qubit_unitary.shape != (
            2**num_qubits,
            2**num_qubits,
        ):
            raise QiskitError(
                f"Gate is not a qubit operation, it has dimensions {qubit_unitary.shape}."
            )

        qutrit_unitary = np.eye(3 ** int(num_qubits), dtype=complex)

        for idx_row in range(qubit_unitary.shape[0]):

            qti_row = convert_basis_index(idx_row)
            for idx_col in range(qubit_unitary.shape[1]):
                qti_col = convert_basis_index(idx_col)

                qutrit_unitary[qti_row, qti_col] = qubit_unitary[idx_row, idx_col]

        return cls(qutrit_unitary, qubit_gate.label if label is None else label)


def convert_basis_index(index: int, basis: int = 3) -> int:
    """Convert the ``index`` from basis-two (qubit) to basis-``basis`` (qudit).

    Args:
        index: The index corresponding to a position in a qubit-based unitary.
        basis: The dimension of the new sub-system.

    Returns:
        The index now mapped to the index position in a unitary matrix whose elements
        have dimension ``basis``.
    """
    return sum(int(bit) * basis**pos for pos, bit in enumerate(bin(index)[2:][::-1]))
