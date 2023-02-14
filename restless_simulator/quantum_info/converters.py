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

from restless_simulator.circuit import QutritQuantumChannelOperation, QutritUnitaryGate


def qudit_circuit_to_super_op(circuit: QuantumCircuit, basis: int = 3) -> SuperOp:
    """Convert a QuantumCircuit to a SuperOp.


    Note that this converter currently only supports qutrit instructions. Future
    work may generalize the qutrit instructions to qudit instructions.

    Args:
        circuit: The quantum circuit to convert to a super operator.
        basis: The number of levels in the wires of the quantum circuit. Note that currently
            only three levels are supported.
    """
    if basis != 3:
        raise NotImplementedError(
            f"Basis other than three are not supported. Got {basis}."
        )

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

        if isinstance(instruction.operation, QutritUnitaryGate):
            # 2) Create a super op for QutritUnitaryGate
            op2 = SuperOp(
                _kraus_to_superop(([instruction.operation.to_matrix()], None)),
                input_dims=(basis,) * len(qargs),
                output_dims=(basis,) * len(qargs),
            )

        elif isinstance(instruction.operation, QutritQuantumChannelOperation):
            # 2) Create a super op for qudit channels
            op2 = instruction.operation.channel

        else:
            raise NotImplementedError(
                f"Only {QutritUnitaryGate.__name__} and {QutritQuantumChannelOperation.__name__}"
                " instructions are supported."
            )

        # 3) Compose the op on the existing one taking the position into account
        full_op = full_op.compose(op2, qargs)

    return full_op
