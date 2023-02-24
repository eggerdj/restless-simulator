# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qutrit Restless Simulator"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV2, Options
from qiskit.qobj import QobjExperimentHeader
from qiskit.quantum_info import DensityMatrix, Kraus
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.transpiler import Target

from restless_simulator.circuit import QutritQuantumChannelOperation, QutritUnitaryGate
from restless_simulator.quantum_info.converters import (
    qudit_circuit_to_super_op,
    circuit_to_qudit_circuit,
)
from .restless_job import RestlessJob
from .sample_buffer import SampleBuffer


# pylint: disable=too-few-public-methods
class RestlessBackendConfiguration:
    """A backend configuration class for :class:`QutritRestlessSimulator`.

    An instance of this class is expected by Qiskit when calling
    :meth:`QutritRestlessSimulator.configuration`.
    """

    # TODO: Subclass `qiskit.providers.models.BackendConfiguration`

    def __init__(self, rep_delay_range=(0, 100e-3)):
        """Creates a restless simulator backend configuration.

        Args:
            rep_delay_range: Possible repetition delays for restless execution in :math:`\\mu{}s`.
                Defaults to (0, 100e-3).
        """
        self.rep_delay_range = rep_delay_range


class RestlessBackendProperties:
    """A backend properties class for :class:`QutritRestlessSimulator`.

    An instance of this class is expected by Qiskit when calling
    :meth:`QutritRestlessSimulator.properties`.
    """

    # TODO: Subclass `qiskit.providers.models.BackendProperties`

    def __init__(self, t1_time=1) -> None:
        """Create a restless backend properties instance.

        Args:
            t1_time: The default T1 time for all qubits/qutrits, in seconds. Defaults to 1.
        """
        self.t1_time = t1_time

    # pylint: disable=unused-argument
    def qubit_property(self, physical_qubit, name: str = None):
        """Returns the property/properties for the given physical qubit.

        The most important qubit property to return is the T1 time, which is set as the same for all
        qubits (i.e., :attr:`t1_time`).

        Args:
            physical_qubit: The physical qubit index.
            name: The name of the qubit property.

        Returns:
            Qubit property or a dictionary of properties if ``name=None``.
        """
        # TODO: Refactor this method to us BackendProperties as this method differs slightly.
        return {"T1": [self.t1_time]}


# pylint: disable=too-many-instance-attributes
@dataclass
class RestlessCircuitData:
    """Class for storing circuit-specific results during restless execution."""

    memory: List[str] = field(default_factory=list)
    """Hex. representation of measurement outcomes."""
    memory_labelled: List[int] = field(default_factory=list)
    """int representation of measurement outcomes."""
    meas_states: List[int] = field(default_factory=list)
    """Collapsed measurement states."""
    post_meas_states: List[int] = field(default_factory=list)
    """Post measurement states."""
    input_states: List[int] = field(default_factory=list)
    """Input states."""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Circuit metadata"""
    channel: QuantumChannel = None
    """Circuit's corresponding quantum channel."""
    transition_matrix: np.ndarray = None
    """Circuit's corresponding transition matrix."""


class QutritRestlessSimulator(BackendV2):
    """A simulator of restless measurements with qutrits.

    Simulate quantum circuits in the qutrit subspace without reset. Each circuit starts running
    immediately after the previous measurement. The restless simulator samples shots by
    building transition matrices. For each circuit a quantum channel describing the circuit
    is created and this quantum channel is turned into a transition matrix.

    This simulator allows one to investigate the effects of leakage in restless circuit
    execution, i.e., circuits are executed without qubit reset. This is typically relevant
    for circuits the perform characterization and calibration tasks.

    .. note::

        This simulator is intended to run for small scale systems (i.e. a few qutrits)
        since it builds the full Hilbert space. The size of the matrices in the simulator
        scale as :math:`3^n` where :math:`n` is the number of wires in the circuit. This
        is sufficient to investigate characterization and calibration experiments which are
        typically executed on a small number of qubits.
    """

    def __init__(self, shots: int = 2048, **kwargs):
        """
        Args:
            shots: The number of shots to simulate which defaults to 2048.
            kwargs: Additional simulator options, set using :meth:`set_options`.
        """
        super().__init__(
            name="QutritRestlessSimulator",
            description="A restless simulator for qutrits.",
        )
        self._setup_target()
        self.set_options(
            shots=shots,
            **kwargs,
        )

    @property
    def shots(self) -> int:
        """Return the number of shots to be used by the simulator."""
        return self.options.shots

    def _setup_target(self):
        """Setup a :class:`Target` instance for the simulator.

        The target has the following features:
        *. Supports only one qubit/qutrit.
        *. Has a pulse sample-time of :math:`0.222ns`.
        *. Supports :class:`QutritUnitaryGate` and :class:`QutritQuantumChannelOperation`
           instructions.
        """
        self._target = Target(
            description="Target for qutrit restless simulator.",
            num_qubits=1,  # TODO: Extend functionality to more than one qutrit.
            dt=0.222e-9,
        )

        # Add dummy gates to indicate support for QutritUnitaryGate and
        # QutritQuantumChannelOperation.
        self._target.add_instruction(QutritUnitaryGate(np.eye(3)), None)
        self._target.add_instruction(
            QutritQuantumChannelOperation(Kraus([np.eye(3)])), None
        )

    @property
    def target(self):
        """The :class:`Target` instance for this simulator."""
        return self._target

    @property
    def max_circuits(self) -> Optional[int]:
        """The maximum number of circuits the simulator can run at once.

        :class:`QutritRestlessSimulator` has no hard-coded limit on the number of circuits that can
        be run.
        """
        return None

    @classmethod
    def _default_options(cls) -> Options:
        """Default options for the restless simulator.

        Options:
            shots: The number of shots to simulate.
            meas_assignment_mat: The measurement assignment matrix to use for all circuits, if not
                set when calling :meth:`run`. Defaults to perfect qubit measurement, where the first
                and second excited states are treated as one excited state. This matrix is the
                matrix for a single qutrit. In multi-qutrit circuits it applies to all qutrits.
            meas_transition_mat: The measurement transition matrix to use for all circuits, if not
                set when calling :meth:`run`. Defaults to an ideal post-measurement process where
                the   measurement state does not change. This matrix is the matrix for a single
                qutrit. In multi-qutrit circuits it applies to all qutrits.
            ignore_measurement_instructions: Whether the simulator should ignore measurement
                instructions in circuits. If False, an error is thrown by the simulator if a
                measurement is encountered.
            compute_cumulative_trans_mats: Whether to return cumulative transition matrices.
                Defaults to True.
            return_meas_state: Whether to return measurement states in results or not. Measurement
                states are the qutrit states prior to post-measurement errors and readout
                assignment. Defaults to False.
            return_post_meas_state: Whether to return post-measurement states in result or not.
                Defaults to False.
            return_memory_labelled: Whether to return labelled memory or not. Defaults to False.
            return_channel: Whether to return circuit channels or not. Defaults to False.
            return_trans_mat: Whether to return circuit transition matrices or not. Defaults to
                False.
        """
        opts = Options(
            shots=1024,
            meas_assignment_mat=np.array([[1, 0, 0], [0, 1, 1]]),
            meas_transition_mat=np.eye(3),
            ignore_measurement_instructions=True,
            compute_cumulative_trans_mats=True,
            return_meas_state=False,
            return_post_meas_state=False,
            return_memory_labelled=False,
            return_channel=False,
            return_trans_mat=False,
        )
        return opts

    @staticmethod
    def compute_circuit_channel(
        in_circuits: List[QuantumCircuit],
    ) -> List[QuantumChannel]:
        """Computes the quantum channels for each circuit.

        Compatible Circuit Instructions:
            * :class:`QutritUnitaryGate`: a qutrit gate represented as a unitary operation.
            * :class:`QutritQuantumChannelOperation`: a qutrit quantum channel, which is converted
              into a SuperOp when used.
            * Any operation that has a ``to_matrix()`` method, as long as the returned NumPy array
              is sa 2x2 or 3x3 matrix.
            * Measurements, which are ignored if the ``ignore_measurement_instructions`` option is
              True. If False, an error is raised.

        Args:
            in_circuits: List of circuits.

        Raises:
            RuntimeError: if an incompatible circuit instruction is encountered.

        Returns:
            A list of quantum channels corresponding to ``in_circs``.
        """
        channels = []

        for circuit in in_circuits:
            # remove final measurements
            circuit_ = circuit.remove_final_measurements(inplace=False)

            qudit_circuit = circuit_to_qudit_circuit(circuit_)
            channels.append(qudit_circuit_to_super_op(qudit_circuit))

        return channels

    def compute_transition_matrices(
        self,
        in_circs: Optional[List[QuantumCircuit]] = None,
        in_channels: Optional[List[QuantumChannel]] = None,
    ) -> List[np.ndarray]:
        r"""Computes transition matrices for circuits or channels.

        Each transition matrix contains the probability :math:`p_j` to get the collapsed measurement
        state :math:`\ket{j}` if the input state to the circuit is :math:`\ket{i}`. These are
        computed by sampling from ``in_channels`` (which can be computed from ``in_circs``). To
        determine the output probabilities :math:`\{p_j\}_{j=0}^2` for a given input state
        :math:`\ket{i}` using a transition matrix :math:`T_k`, the following operation is carried
        out

        .. math::

            \vec{O} = (p_0,p_1,p_2)^T = T_k\vec{I}

        where :math:`\vec{I} = (a_0,a_1,a_2)^T` is the input state subject to :math:`a_i = 1` and
        :math:`a_{k\neq{}i} = 0`.

        .. warning:
            This code currently only supports one qutrit. Eventually this will be extended to
            handle multi-qutrit circuits and channels.

        Args:
            in_circs (_type_, optional): The circuits for which the transition matrices must be
                computed, by first computing their corresponding channel representations. If
                ``in_channels`` is provided, ``in_circs`` is ignored. Defaults to None.
            in_channels (_type_, optional): Optional list of channels for which the transition
                matrices must be computed. If None, ``in_circs`` is first converted into channels
                before the transition matrices are computed. Defaults to None.

        Raises:
            QiskitError: if neither ``in_circs`` or ``in_channels`` are set.

        Returns:
            List of transition matrices corresponding to the input circuits/channels.
        """
        if in_channels is None and in_circs is None:
            raise QiskitError(
                "one of in_circs and in_channels must be set, but both are None."
            )

        if in_circs is not None and in_channels is not None:
            warnings.warn(
                "in_circs and in_channels are both set, but in_circs is ignored."
            )

        if in_channels is None:
            in_channels = self.compute_circuit_channel(in_circuits=in_circs)

        transition_matrices = []

        for channel in in_channels:
            # Number of states is the product of the sub-dimensions.
            n_states = np.prod(channel.input_dims())

            transition_matrix = np.zeros((n_states, n_states))
            for i in range(n_states):
                # Create input state as a single qutrit pure-state: |0>, |1>, or |2>.
                input_state_mat = np.zeros((n_states,))
                input_state_mat[i] = 1
                input_state_mat = np.diag(input_state_mat)
                input_state = DensityMatrix(input_state_mat)

                # Compute the output state-vector
                output_state = input_state.evolve(channel)
                transition_matrix[:, i] = output_state.probabilities()

            transition_matrices.append(transition_matrix)

        return transition_matrices

    def get_sample_buffers(
        self, trans_mats: List[np.ndarray], size: int = 1024
    ) -> List[SampleBuffer]:
        """Create a list of sample buffers from transition matrices.

        Each sample buffer contains presampled outcomes for the given transition matrices, to assist
        with speeding up simulation of circuits. Sample buffers reduce the number of calls to
        :func:`numpy.random.choice`, which is costly when simulating multiple circuits with at least
        1000 shots.

        :class:`SampleBuffer` samples a transition matrix (``trans_mats``) with :math:`3n` calls to
        :func:`numpy.random.choice`, where :math:`n` is the number of qutrits. In each call,
        ``size`` many samples are taken. Without a sample buffer, there would be one matrix
        multiplication step and one call to :func:`numpy.random.choice` per sample of a given
        transition matrix. For more detail on how the sample buffers are implemented, see the
        documentation for :class:`SampleBuffer`.

        Args:
            trans_mats: List of transition matrices.
            size: Number of samples to buffer.

        Returns:
            List of sample buffers for each transition matrix.
        """
        return [SampleBuffer(t_mat, size=size) for t_mat in trans_mats]

    @classmethod
    def _memory_to_counts(cls, memory: Sequence[Union[int, str]]) -> Dict[str, int]:
        """Convert a list of single-shot states to a counts dictionary.

        Args:
            memory: List of single-shot labels.

        Returns:
            A counts dictionary for the given input memory list.
        """
        unique_states, num_counts = np.unique(memory, return_counts=True)

        return dict(zip(unique_states, num_counts))

    list_union_array = Union[np.ndarray, List[np.ndarray]]
    """Typing alias for a numpy array or list of numpy arrays."""

    def _validated_mats(
        self,
        circuits: List[QuantumCircuit],
        meas_assignment_mats: Optional[list_union_array] = None,
        meas_transition_mats: Optional[list_union_array] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Create validated lists of measurement assignment and transition matrices.

        This method confirms that ``meas_assignment_mats`` and ``meas_transition_mats`` are valid
        (i.e., either a NumPy array of the correct shape or a list of equivalent NumPy arrays). If
        single NumPy arrays are given, lists of ``n_circuits`` copies are created. If lists of
        arrays are given, then the number of entries must be equal to ``n_circuits``. If ``None`` is
        passed as the value for either input matrix arguments, their corresponding default value
        from the simulator options is used.

        If any of these requirements are not met, an :class:`AttributeError` is raised.

        Args:
            circuits: The quantum circuits to simulate.
            meas_assignment_mats: Optional list of measurement assignment matrices. If None, the
                ``meas_assignment_mat`` option is used for all circuits. Defaults to None.
            meas_transition_mats: Optional list of post-measurement transition matrices. If None,
                the ``meas_transition_mat`` option is used for all circuits. Defaults to None.

        Raises:
            AttributeError: if the number of entries in the list ``meas_assignment_mats`` is not
                equal to ``n_circuits``.
            AttributeError: if the number of entries in the list ``meas_transition_mats`` is not
                equal to ``n_circuits``.

        Returns:
            The tuple ``(assignment, transition)`` of lists containing ``n_circuits`` entries
            corresponding to the input arguments ``meas_assignment_mats`` and
            ``meas_transition_mats`` respectively.
        """
        n_circuits = len(circuits)

        # Handle single array for all circuits for `meas_assignment_mats` and `meas_transition_mats`
        if isinstance(
            meas_assignment_mats, np.ndarray
        ) and meas_assignment_mats.shape in [
            (3, 3),
            (2, 3),
        ]:
            meas_assignment_mats = [meas_assignment_mats] * n_circuits
        if isinstance(
            meas_transition_mats, np.ndarray
        ) and meas_transition_mats.shape == (3, 3):
            meas_transition_mats = [meas_transition_mats] * n_circuits

        # Check lengths of `meas_assignment_mats` and `meas_transition_mats`.
        if meas_assignment_mats is not None and len(meas_assignment_mats) != n_circuits:
            raise AttributeError(
                "Length of meas_assignment_mats doesn't match length of circuits: expected "
                f"{n_circuits} entries but got {len(meas_assignment_mats)}."
            )
        if meas_transition_mats is not None and len(meas_transition_mats) != n_circuits:
            raise AttributeError(
                "Length of meas_transition_mats doesn't match length of circuits: expected "
                f"{n_circuits} entries but got {len(meas_transition_mats)}."
            )

        # Set defaults for `meas_assignment_mats` and `meas_transition_mats` if necessary.
        if meas_assignment_mats is None:
            meas_assignment_mats = [self.options.meas_assignment_mat] * n_circuits
        if meas_transition_mats is None:
            meas_transition_mats = [self.options.meas_transition_mat] * n_circuits

        # Expand the matrices to the full Hilbert space size.
        n_qutrits = len(circuits[0].qregs[0])
        meas_assignment_mats = [
            self._expand_matrix(mat, n_qutrits) for mat in meas_assignment_mats
        ]
        meas_transition_mats = [
            self._expand_matrix(mat, n_qutrits) for mat in meas_transition_mats
        ]

        return meas_assignment_mats, meas_transition_mats

    @staticmethod
    def _expand_matrix(matrix, num_elements):
        """expands a matrix to the full Hilbert space."""
        if num_elements == 1:
            return matrix

        full_matrix = np.copy(matrix)
        for _ in range(num_elements - 1):
            full_matrix = np.kron(full_matrix, matrix)

        return full_matrix

    def _create_experiment_results(
        self,
        circuit_data: List[RestlessCircuitData],
        circuits: List[QuantumCircuit],
        **kwargs,
    ) -> List[ExperimentResult]:
        experiment_results = []
        for data, circ in zip(circuit_data, circuits):
            exp_data = ExperimentResultData(
                memory=data.memory, counts=self._memory_to_counts(data.memory)
            )

            # Get circuit metadata if it exists.
            if circ.metadata is not None:
                circ_metadata = circ.metadata
            else:
                circ_metadata = {}

            n_wires = len(circuits[0].qregs[0])
            header = QobjExperimentHeader(
                metadata=circ_metadata,
                n_qubits=n_wires,
                memory_slots=n_wires,
                creg_sizes=[["meas", n_wires]],
            )

            ## Add optional outputs depending on return_X options.
            result_data: Dict[str, Any] = {}
            # return_post_meas_state
            if kwargs.get(
                "return_post_meas_state", self.options.return_post_meas_state
            ):
                result_data["post_meas_state"] = data.post_meas_states

            # return_meas_state
            if kwargs.get("return_post_meas_state", self.options.return_meas_state):
                result_data["meas_state"] = data.meas_states

            # return_memory_labelled
            if kwargs.get(
                "return_memory_labelled", self.options.return_memory_labelled
            ):
                result_data["memory_labelled"] = data.memory_labelled

            # return_channel
            if kwargs.get("return_channel", self.options.return_channel):
                result_data["channel"] = data.channel

            # return_trans_mat
            if kwargs.get("return_trans_mat", self.options.return_trans_mat):
                result_data["trans_mat"] = data.transition_matrix
            ##

            ## Create experiment result instance and add to list
            exp_res = ExperimentResult(
                shots=self.options.shots,
                success=True,
                data=exp_data,
                header=header,
                meas_level=1,
                status="Success",
                meas_return="single",
                **result_data,
            )
            experiment_results.append(exp_res)
            ##
        return experiment_results

    def _create_job(
        self,
        circuit_data: List[RestlessCircuitData],
        circuits: List[QuantumCircuit],
        cum_trans_mats: List[np.ndarray],
        **kwargs,
    ) -> RestlessJob:
        ## Create result objects
        experiment_results = self._create_experiment_results(
            circuit_data, circuits, **kwargs
        )
        ##

        ## Create and return RestlessJob
        job_id = str(uuid.uuid4())
        # `cum_trans_mats` is a result for the entire job and not an individual circuit, so we add
        # it to the restless job and not an experiment result.
        job_kwargs: Dict[str, Any] = {
            "job_id": job_id,
            # Default is None so we at least define the attribute in the result.
            "cum_trans_mats": None,
        }
        if self.options.compute_cumulative_trans_mats:
            job_kwargs["cum_trans_mats"] = cum_trans_mats
        return RestlessJob(
            self,
            job_id=job_id,
            result=Result(
                backend_name=self.name,
                backend_version=self.backend_version,
                qobj_id="RESTLESS_CIRCUIT",  # Dummy qobj_id
                success=True,
                results=experiment_results,
                **job_kwargs,
            ),
        )
        ##

    def _init_circuit_data(
        self, circuits: List[QuantumCircuit]
    ) -> List[RestlessCircuitData]:
        """Initialize the circuit data.

        The circuit data holds the quantum channel for each circuit as well as
        its transition matrix.

        Args:
            circuits: The circuits for which to initialize the circuit data.

        Raises:
            QiskitError: If the circuits do not all have the same number of wires.
        """
        n_wires = len(circuits[0].qregs[0])
        for circuit in circuits:
            if len(circuit.qregs[0]) != n_wires:
                raise QiskitError(
                    "The restless simulator only accepts circuits with the same number of wires."
                )

        channels = self.compute_circuit_channel(in_circuits=circuits)
        transition_matrices = self.compute_transition_matrices(in_channels=channels)
        circuit_data = [
            RestlessCircuitData(
                metadata=circ.metadata, channel=channel, transition_matrix=trans_mat
            )
            for circ, channel, trans_mat in zip(circuits, channels, transition_matrices)
        ]
        return circuit_data

    @staticmethod
    def _simulate_single_shot(
        input_state: int,
        circuit_data: RestlessCircuitData,
        circuit_buffer: SampleBuffer,
        meas_assign_buffer: SampleBuffer,
        post_meas_buffer: SampleBuffer,
    ) -> int:
        """Simulate a circuit to generate restless results and update the circuit data instance.

        This method generates restless results for a single shot of the circuit corresponding to
        ``circuit_data``. The circuit has associated buffers for the collapsed measurement state
        (``circuit_buffer``), measurement outcome post measurement assignment
        (``meas_assign_buffer``), and the post-measurement state (``post_meas_buffer``).

        The returned value is the sampled post-measurement state (``int``), which is the input state
        for the next restless shot.

        Args:
            input_state: The input state to use when sampling from the circuit buffer.
            circuit_data: The :class:`RestlessCircuitData` instance for the circuit being
                simulated/sampled.
            circuit_buffer: The circuit's corresponding collapsed measurement state buffer.
            meas_assign_buffer: The circuit's corresponding measurement outcome buffer.
            post_meas_buffer: The circuit's corresponding post-measurement state buffer.

        Returns:
            The post-measurement state label, which is the input to the next circuit with restless
            circuit execution.
        """
        # 1. Get probabilities to measure |0>, |1>, or |2> for the given input state
        meas_state = circuit_buffer.get_label(input_state)

        # 2.a Apply assignment matrix
        measurement = meas_assign_buffer.get_label(meas_state)

        # 2.b Apply post-measurement transition matrix
        post_meas_state = post_meas_buffer.get_label(meas_state)

        # 3. Store results of circuit simulation
        circuit_data.memory.append(hex(measurement))
        circuit_data.memory_labelled.append(measurement)
        circuit_data.meas_states.append(meas_state)
        circuit_data.post_meas_states.append(post_meas_state)

        return post_meas_state

    # pylint: disable=arguments-renamed
    def run(
        self,
        circuits: Union[List[QuantumCircuit], QuantumCircuit],
        meas_assignment_mats: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        meas_transition_mats: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        **kwargs,
    ) -> RestlessJob:
        """Simulate qutrit circuits with restless circuit execution.

        Args:
            circuits: A circuit or list of circuits, with mixed qutrit and qubit operations.
                Measurement gates are not currently supported, and are implicitly added during
                simulation.
            meas_assignment_mats: Optional list of measurement assignment matrices. If None, the
                ``meas_assignment_mat`` option is used for all circuits. Defaults to None. This
                argument can be a list so that each circuit can have its unique assignment matrix.
            meas_transition_mats: Optional list of post-measurement transition matrices. If None,
                the ``meas_transition_mat`` option is used for all circuits. Defaults to None. This
                argument can be a list so that each circuit can have its unique measurement
                transition matrix.

        Raises:
            AttributeError: if the number of measurement assignment matrices doesn't match the
                number of circuits.
            AttributeError: if the number of post-measurement transition matrices doesn't match the
                number of circuits.

        Returns:
            A job with the results of simulating all circuits in a restless manner.
        """
        self.set_options(**kwargs)

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        # Create circuit sample buffers and circuit_data list.
        # `circuit_data` is a list of RestlessCircuitData instances which contain restless results
        # for each circuit.
        circuit_data = self._init_circuit_data(circuits)

        # `circuit_buffers` is a list of sample buffers for the collapsed measurement states of each
        # circuit.
        circuit_buffers = self.get_sample_buffers(
            trans_mats=[circ_data.transition_matrix for circ_data in circuit_data],
            size=self.options.shots,
        )
        ##

        ## Create measurement sample buffers.
        # Validate input measurement arguments.
        meas_assignment_mats, meas_transition_mats = self._validated_mats(
            circuits, meas_assignment_mats, meas_transition_mats
        )

        # Create buffers.
        meas_assign_buffers = self.get_sample_buffers(
            trans_mats=meas_assignment_mats, size=self.options.shots
        )
        post_meas_buffers = self.get_sample_buffers(
            trans_mats=meas_transition_mats, size=self.options.shots
        )
        ##

        # Create list of cumulative transition matrices. Only used if
        # self.options.compute_cumulative_trans_mats is True.
        cum_trans_mats = []

        # Start with the ground-state |0>
        prev_state = 0
        # Previous shot cumulative transition matrix. Will be initialized to ID.
        prev_trans_mat = None

        # Loop over shots
        for _ in range(self.options.shots):
            # Loop over circuit transition matrices and post-measurement transition matrices.
            for i_circ, circ_data in enumerate(circuit_data):
                # Add input states to memory
                circuit_data[i_circ].input_states.append(prev_state)

                # Compute cumulative transition matrix
                if self.options.compute_cumulative_trans_mats:
                    if prev_trans_mat is None:
                        prev_trans_mat = np.eye(circ_data.transition_matrix.shape[0])

                    curr_cum_trans_mat = circ_data.transition_matrix @ prev_trans_mat
                    cum_trans_mats.append(curr_cum_trans_mat.copy())
                    prev_trans_mat = meas_transition_mats[i_circ] @ curr_cum_trans_mat

                # Compute results, update circuit_data instance, and update prev_state for next
                # shot.
                prev_state = self._simulate_single_shot(
                    input_state=prev_state,
                    circuit_data=circ_data,
                    circuit_buffer=circuit_buffers[i_circ],
                    meas_assign_buffer=meas_assign_buffers[i_circ],
                    post_meas_buffer=post_meas_buffers[i_circ],
                )

        return self._create_job(circuit_data, circuits, cum_trans_mats, **kwargs)

    def configuration(self) -> RestlessBackendConfiguration:
        """Return the backend configuration."""
        return RestlessBackendConfiguration()

    def properties(self) -> RestlessBackendProperties:
        """Return the backend properties."""
        return RestlessBackendProperties()
