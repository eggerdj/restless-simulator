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

from qiskit.circuit import QuantumCircuit, Gate
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


def circuit_to_qudit_circuit(
    circuit: QuantumCircuit, inplace: bool = False
) -> QuantumCircuit:
    """Return a circuit where all instructions are embedded in a larger space.

    This function converts a qubit-based circuit to one where all instructions are embedded
    in a larger hilbert space. This is intended for simulation purposes, i.e., the new
    instructions have a to-matrix method that can be used by ``qudit_circuit_to_super_op``.

    Args:
        circuit: The circuit to convert.
        inplace: If true then the given ``circuit`` is modified inplace. If false a modified
            copy of the circuit is returned.

    Returns:
        A quantum circuit where all ``Gate`` instructions have been embedded in a larger space.
    """
    new_circuit = circuit if inplace else circuit.copy()

    for idx, inst in enumerate(new_circuit.data):
        if isinstance(
            inst.operation, (QutritUnitaryGate, QutritQuantumChannelOperation)
        ):
            continue
        if isinstance(inst.operation, Gate):
            new_circuit.data[idx].operation = QutritUnitaryGate.from_qubit_gate(
                inst.operation,
                label=inst.operation.name,
            )
        else:
            raise NotImplementedError(
                f"Can only convert instances of Gate to {QutritUnitaryGate.__name__}."
            )

    return new_circuit
