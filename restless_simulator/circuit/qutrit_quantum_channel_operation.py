"""Qutrit compatible operation to store QuantumChannels in QuantumCircuits."""
from typing import Optional

import numpy as np
from qiskit.circuit import Operation
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel


class QutritQuantumChannelOperation(Operation):
    """Custom circuit operation class to allow for quantum channels on qutrits."""

    def __init__(
        self,
        channel: QuantumChannel,
        name: str = "Channel",
        label: Optional[str] = None,
    ):
        """Create a qutrit quantum-channel operation.

        Args:
            channel: The QuantumChannel for the operation. Must be for qutrits.
            name: Optional name of the operation. Defaults to "Channel".
            label: Optional label of the operation. Defaults to None.

        Raises:
            QiskitError: If the channel is not an N-qutrit channel.
            QiskitError: If the channel is not CPTP.
        """
        super().__init__()
        self._channel = channel

        n_qutrits = int(np.log2(self._channel.input_dims()) / np.log2(3))
        if (
            self._channel.input_dims() != self._channel.output_dims()
            or 3**n_qutrits != self._channel.input_dims()[0]
        ):
            raise QiskitError(
                f"Cannot convert QuantumChannel to {self.__class__.__name__}: channel is not an "
                r"N-qutrit channel.\n"
                f"{self._channel.input_dims()}-{self._channel.output_dims()}"
            )
        if not self._channel.is_cptp():
            raise QiskitError(
                "Cannot convert QuantumChannel to {self.__class__.__name__}: channel "
                "is not CPTP."
            )
        self._name = name
        self._label = label if label is not None else name

    @property
    def num_qubits(self):
        """Number of qutrits.

        Qiskit assumes qubits and thus :class:`QutritQuantumChannelOperation` inherits
        :meth:`num_qubits` from :class:`Operation`.
        """
        return int(np.log2(self._channel.input_dims()[0]) / np.log2(3))

    @property
    def num_qutrits(self):
        """Number of qutrits."""
        return self.num_qubits

    @property
    def num_clbits(self):
        """QutritQuantumChannelOperation is restricted to purely quantum channels."""
        return 0

    @property
    def name(self):
        """Name of the operation."""
        return self._name

    @property
    def condition(self):
        """Returns None so that :meth:`QuantumCircuit.draw` doesn't raise an error."""
        return None

    @property
    def channel(self):
        """Returns the underlying quantum channel objective for this operation.

        Returns:
            QuantumChannel: The quantum channel object for this operation.
        """
        return self._channel

    def __repr__(self) -> str:
        return f'{type(self).__name__}(channel={self.channel},name="{self.name}")'

    @property
    def label(self) -> str:
        """Label of the quantum channel operation."""
        return self._label

    @label.setter
    def label(self, name: str):
        self._label = name

    def copy(self) -> "QutritQuantumChannelOperation":
        """Returns a copy of this qutrit quantum channel operation.

        Returns:
            QutritQuantumChannelOperation: A copy of this operation instance.
        """
        return QutritQuantumChannelOperation(self.channel, self.name, self.label)
