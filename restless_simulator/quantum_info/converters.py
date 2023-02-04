# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Convert qudit-based circuits to a SuperOp."""

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SuperOp


def qudit_circuit_to_super_op(circuit: QuantumCircuit, dim: int = 3) -> SuperOp:
    """Convert a QuantumCircuit or Instruction to a SuperOp.

    The instructions in the quantum circuit need to support a ``to_matrix`` method
    that results in matrices of the correct dimension.

    Args:
        circuit: The quantum circuit to convert to a SuperOp instance.
        dim: The dimension of the wires in the quantum circuit.

    Returns:
        A super operator in which all sub-systems have dimension ``dim``.
    """
    instruction = circuit.to_instruction()

    n_sys = instruction.num_qubits
    dims = (dim, ) * n_sys
    op = SuperOp(np.eye((dim * dim)**n_sys), input_dims=dims, output_dims=dims)
    op._append_instruction(instruction)

    return op
