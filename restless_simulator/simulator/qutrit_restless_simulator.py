"""Qutrit Restless Simulator"""
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV2, Options
from qiskit.qobj import QobjExperimentHeader
from qiskit.quantum_info import DensityMatrix, Kraus, SuperOp
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.transpiler import Target

from restless_simulator.circuit import QutritQuantumChannelOperation, QutritUnitaryGate
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
            rep_delay_range: Possible repetition delays for restless execution. Defaults to (0,
                100e-3).
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
class CircuitData:
    """Class for storing circuit-specific results during restless execution."""

    memory: List[str] = []
    """Hex. representation of measurement outcomes."""
    memory_labelled: List[int] = []
    """int representation of measurement outcomes."""
    meas_states: List[int] = []
    """Collapsed measurement states."""
    post_meas_states: List[int] = []
    """Post measurement states."""
    input_states: List[int] = []
    """Input states."""
    metadata: Dict[str, Any] = {}
    """Circuit metadata"""
    channel: QuantumChannel = None
    """Circuit's corresponding quantum channel."""
    transition_matrix: np.ndarray = None
    """Circuit's corresponding transition matrix."""


class QutritRestlessSimulator(BackendV2):
    """A simulator of restless measurements with qutrits."""

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
                and second excited states are treated as one excited state.
            meas_transition_mat: The measurement transition matrix to use for all circuits, if not
                set when calling :meth:`run`. Defaults to an ideal post-measurement process where
                the   measurement state does not change.
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

    def compute_circuit_channel(
        self, in_circs: List[QuantumCircuit]
    ) -> List[QuantumChannel]:
        """Computes the quantum channels for each circuit.

        Compatible Circuit Instructions:
            *. :class:`QutritUnitaryGate`: a qutrit gate represented as a unitary operation.
            *. :class:`QutritQuantumChannelOperation`: a qutrit quantum channel, which is converted
               into a SuperOp when used.
            *. Any operation that has a `to_matrix()` method, as long as the returned NumPy array is
               sa 2x2 or 3x3 matrix.
            *. Measurements, which are ignored if the ``ignore_measurement_instructions`` option is
               True. If False, an error is raised.

        Args:
            in_circs: List of circuits.

        Raises:
            RuntimeError: if an incompatible circuit instruction is encountered.

        Returns:
            A list of quantum channels corresponding to ``in_circs``.
        """
        channels = []
        for circ in in_circs:
            # Decompose in-case we have nested circuits
            simplified_circ = circ.decompose()
            # Create initial identity channel
            channel = SuperOp(np.eye(9).astype(np.float128))
            for inst in simplified_circ.data:
                # QutritUnitaryGate
                if isinstance(inst.operation, QutritUnitaryGate):
                    operator = inst.operation.as_operator()
                    inst_channel = SuperOp(Kraus(operator.data))
                    channel = inst_channel @ channel
                # QutritQuantumChannelOperation
                elif isinstance(inst.operation, QutritQuantumChannelOperation):
                    inst_channel = inst.operation.channel
                    channel = SuperOp(inst_channel) @ channel
                # If operation has a `to_matrix()` method
                elif hasattr(inst.operation, "to_matrix"):
                    mat = inst.operation.to_matrix()
                    if mat.shape == (3, 3):
                        inst_channel = SuperOp(Kraus(mat))
                    elif mat.shape == (2, 2):
                        qutrit_mat = np.eye(3, dtype=complex)
                        qutrit_mat[0:2, 0:2] = mat
                        inst_channel = SuperOp(Kraus(qutrit_mat))
                    else:
                        raise RuntimeError(
                            f"{type(self).__name__} encountered an instruction with a "
                            "'to_matrix()' method but the shape of the matrix is not supported: "
                            f"expected (3,3) or (2,2) but got {mat.shape}."
                        )
                    channel = inst_channel @ channel
                elif (
                    inst.operation.name == "measure"
                    and self.options.ignore_measurement_instructions
                ):
                    continue
                else:
                    raise RuntimeError(
                        f"{self.__class__.__name__} encountered unknown instruction of type "
                        f"{type(inst.operation).__name__}: {inst.operation}"
                    )
            channels.append(channel)
        return channels

    def compute_transition_matrices(
        self,
        in_circs: Optional[List[QuantumCircuit]] = None,
        in_channels: Optional[List[QuantumChannel]] = None,
    ) -> List[np.ndarray]:
        """Computes transition matrices for circuits or channels.

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
            in_channels = self.compute_circuit_channel(in_circs=in_circs)

        transition_matrices = []

        # Variable to control the number of states. For 1 qutrit, this is three. We assume that all
        # channels and circuits are for one qutrit.
        # TODO: Extend this code to handle multiple qutrits.
        n_states = 3
        for chann in in_channels:
            transition_matrix = np.zeros((n_states, n_states))
            for i in range(n_states):
                # Create input state as a single qutrit pure-state: |0>, |1>, or |2>.
                input_state_mat = np.zeros((n_states,))
                input_state_mat[i] = 1
                input_state_mat = np.diag(input_state_mat)
                input_state = DensityMatrix(input_state_mat)

                # Compute the output statevector
                output_state = input_state.evolve(chann)
                transition_matrix[:, i] = output_state.probabilities()
            transition_matrices.append(transition_matrix)
        return transition_matrices

    def get_sample_buffers(
        self, trans_mats: List[np.ndarray], size: int = 1024
    ) -> List[SampleBuffer]:
        """Create a list of sample buffers from transition matrices.

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
        n_circuits: int,
        meas_assignment_mats: Optional[list_union_array] = None,
        meas_transition_mats: Optional[list_union_array] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Create validated lists of measurement assignment and transition matrices, or raise an
        error.

        This method confirms that ``meas_assignment_mats`` and ``meas_transition_mats`` are valid
        (i.e., either a NumPy array of the correct shape or a list of equivalent NumPy arrays). If
        single NumPy arrays are given, lists of ``n_circuits`` copies are created. If lists of
        arrays are given, then the number of entries must be equal to ``n_circuits``. If ``None`` is
        passed as the value for either input matrix arguments, their corresponding default value
        from the simulator options is used.

        If any of these requirements are not met, an :class:`AttributeError` is raised.

        Args:
            n_circuits: Number of circuits.
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
        ## Check input arguments
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
        if not meas_assignment_mats is None:
            meas_assignment_mats = [self.options.meas_assignment_mat] * n_circuits
        if not meas_transition_mats is None:
            meas_transition_mats = [self.options.meas_transition_mat] * n_circuits
        ##
        return (meas_assignment_mats, meas_transition_mats)

    def _create_experiment_results(
        self,
        circuit_data: List[CircuitData],
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
            header = QobjExperimentHeader(metadata=circ_metadata)

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
        circuit_data: List[CircuitData],
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

    def _initialize_circuit_data(
        self, circuits: List[QuantumCircuit]
    ) -> List[CircuitData]:
        channels = self.compute_circuit_channel(in_circs=circuits)
        transition_matrices = self.compute_transition_matrices(in_channels=channels)
        circuit_data = [
            CircuitData(
                metadata=circ.metadata, channel=channel, transition_matrix=trans_mat
            )
            for circ, channel, trans_mat in zip(circuits, channels, transition_matrices)
        ]
        return circuit_data

    # pylint: disable=too-many-arguments
    def _simulate_single_shot(
        self,
        input_state: int,
        circuit_data: CircuitData,
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
            circuit_data: The :class:`CircuitData` instance for the circuit being simulated/sampled.
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
        circuits: List[QuantumCircuit],
        meas_assignment_mats: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        meas_transition_mats: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        **kwargs,
    ) -> RestlessJob:
        """Simulate qutrit circuits with restless circuit execution.

        Args:
            circuits: List of circuits, with mixed qutrit and qubit operations. Measurement gates
                are not currently supported, and are implicitly added during simulation.
            meas_assignment_mats: Optional list of measurement assignment matrices. If None, the
                ``meas_assignment_mat`` option is used for all circuits. Defaults to None.
            meas_transition_mats: Optional list of post-measurement transition matrices. If None,
                the ``meas_transition_mat`` option is used for all circuits. Defaults to None.

        Raises:
            AttributeError: if the number of measurement assignment matrices doesn't match the
                number of circuits.
            AttributeError: if the number of post-measurement transition matrices doesn't match the
                number of circuits.

        Returns:
            A job with the results of simulating all circuits in a restless manner.
        """

        ## Create circuit sample buffers and circuit_data list.
        # `circuit_data` is a list of CircuitData instances which contain restless results for each
        # circuit.
        circuit_data = self._initialize_circuit_data(circuits)

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
            len(circuits), meas_assignment_mats, meas_transition_mats
        )

        # Create buffers.
        meas_assign_buffers = self.get_sample_buffers(
            trans_mats=meas_assignment_mats, size=self.options.shots
        )
        post_meas_buffers = self.get_sample_buffers(
            trans_mats=meas_transition_mats, size=self.options.shots
        )
        ##

        ## Create list of cumulative transition matrices. Only used if
        ## self.options.compute_cumulative_trans_mats is True.
        cum_trans_mats = []
        ##

        # Start with the ground-state |0>
        prev_state = 0
        # Previous shot cumulative transition matrix.
        prev_trans_mat = np.eye(3).astype(np.float128)

        # Loop over shots
        for _ in range(self.options.shots):
            # Loop over circuit transition matrices and post-measurement transition matrices.
            for i_circ, circ_data in enumerate(circuit_data):
                # Add input states to memory
                circuit_data[i_circ].input_states.append(prev_state)

                # Compute cumulative transition matrix
                if self.options.compute_cumulative_trans_mats:
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
