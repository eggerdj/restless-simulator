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
from qiskit.quantum_info.operators.channel.transformations import _kraus_to_superop


def qudit_circuit_to_super_op(circuit: QuantumCircuit, basis: int = 3) -> SuperOp:
    """Convert a QuantumCircuit or Instruction to a SuperOp."""

    n_sys = circuit.num_qubits
    full_op = SuperOp(
        np.eye((basis**2) ** n_sys),
        input_dims=(basis,) * n_sys,
        output_dims=(basis,) * n_sys,
    )

    qreg = circuit.qregs[0]  # Qubit register
    for instruction in circuit.data:
        # 1) get qargs, i.e. wires on which the instruction applies.
        qargs = [qreg.index(qubit) for qubit in instruction.qubits]

        # 2) Create a super op for this instruction
        op2 = SuperOp(
            _kraus_to_superop(([instruction.operation.to_matrix()], None)),
            input_dims=(basis,) * len(qargs),
            output_dims=(basis,) * len(qargs),
        )

        # 3) Compose the op on the existing one taking the position into account
        full_op.compose(op2, qargs)

    return full_op
