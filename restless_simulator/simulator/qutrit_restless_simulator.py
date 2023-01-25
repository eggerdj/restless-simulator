"""Qutrit Restless Simulator"""
import uuid
from typing import Dict, List, Optional, Union
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
                input_state = np.zeros((n_states,))
                input_state[i] = 1
                input_state = np.diag(input_state)
                input_state = DensityMatrix(input_state)

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
    def _memory_to_counts(cls, memory: List[Union[int, str]]) -> Dict[str, int]:
        """Convert a list of single-shot states to a counts dictionary.

        Args:
            memory: List of single-shot labels.

        Returns:
            A counts dictionary for the given input memory list.
        """
        unique_states, num_counts = np.unique(memory, return_counts=True)

        return dict(zip(unique_states, num_counts))

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
        ## Check input arguments
        # Handle single array for all circuits for `meas_assignment_mats` and `meas_transition_mats`
        if isinstance(
            meas_assignment_mats, np.ndarray
        ) and meas_assignment_mats.shape in [
            (3, 3),
            (2, 3),
        ]:
            meas_assignment_mats = [meas_assignment_mats] * len(circuits)
        if isinstance(
            meas_transition_mats, np.ndarray
        ) and meas_transition_mats.shape == (3, 3):
            meas_transition_mats = [meas_transition_mats] * len(circuits)

        # Check lengths of `meas_assignment_mats` and `meas_transition_mats`.
        if meas_assignment_mats and len(meas_assignment_mats) != len(circuits):
            raise AttributeError(
                "Length of meas_assignment_mats doesn't match length of circuits: expected "
                f"{len(circuits)} but got {len(meas_assignment_mats)}."
            )
        if meas_transition_mats and len(meas_transition_mats) != len(circuits):
            raise AttributeError(
                "Length of meas_transition_mats doesn't match length of circuits: expected "
                f"{len(circuits)} but got {len(meas_transition_mats)}."
            )

        # Set defaults for meas_assignment_mats` and `meas_transition_mats` if necessary.
        if not meas_assignment_mats:
            meas_assignment_mats = [self.options.meas_assignment_mat] * len(circuits)
        if not meas_transition_mats:
            meas_transition_mats = [self.options.meas_transition_mat] * len(circuits)
        ##

        # Compute channels
        channels = self.compute_circuit_channel(circuits)

        # Compute transition matrices
        trans_mats = self.compute_transition_matrices(in_channels=channels)

        # Compute circuit sample buffers, which store possible outputs instead of computing them for
        # each shot.
        sample_buffers = self.get_sample_buffers(
            trans_mats=trans_mats, size=self.options.shots
        )
        meas_assign_buffers = self.get_sample_buffers(
            trans_mats=meas_assignment_mats, size=self.options.shots
        )
        post_meas_buffers = self.get_sample_buffers(
            trans_mats=meas_transition_mats, size=self.options.shots
        )

        ## Compute counts
        # Create lists to contain simulation results
        cum_trans_mats = []
        sorted_memory = [
            {
                "memory": [],
                "memory_labelled": [],
                "meas_state": [],
                "post_meas_state": [],
                "input_states": [],
                "metadata": circ.metadata,
                "channel": channel,
                "transition_matrix": trans_mat,
            }
            for circ, channel, trans_mat in zip(circuits, channels, trans_mats)
        ]

        # Start with the ground-state |0>
        prev_state = 0
        # Previous shot cumulative transition matrix.
        prev_trans_mat = np.eye(3).astype(np.float128)

        # Loop over shots
        for _ in range(self.options.shots):
            # Loop over circuit transition matrices and post-measurement transition matrices.
            for i_circ, (
                circ_trans_mat,
                meas_trans_mat,
            ) in enumerate(zip(trans_mats, meas_transition_mats)):
                # Add input states to memory
                sorted_memory[i_circ]["input_states"].append(prev_state)

                # Compute cumulative transition matrix
                if self.options.compute_cumulative_trans_mats:
                    curr_cum_trans_mat = circ_trans_mat @ prev_trans_mat
                    cum_trans_mats.append(curr_cum_trans_mat.copy())
                    prev_trans_mat = meas_trans_mat @ curr_cum_trans_mat

                # 1. Get probabilities to measure |0>, |1>, or |2> for the previous state
                meas_state = sample_buffers[i_circ].get_label(prev_state)

                # 2.a Apply assignment matrix
                measurement = meas_assign_buffers[i_circ].get_label(meas_state)

                # 2.b Apply post-measurement transition matrix
                post_meas_state = post_meas_buffers[i_circ].get_label(meas_state)

                # 3. Store results of circuit and set previous state as input to next shot
                prev_state = post_meas_state

                # Add results to sorted memory
                sorted_memory[i_circ]["memory"].append(hex(measurement))
                sorted_memory[i_circ]["memory_labelled"].append(measurement)
                sorted_memory[i_circ]["meas_state"].append(meas_state)
                sorted_memory[i_circ]["post_meas_state"].append(post_meas_state)

        ## Create result objects
        experiment_results = []
        for data, circ in zip(sorted_memory, circuits):
            exp_data = ExperimentResultData(
                memory=data["memory"], counts=self._memory_to_counts(data["memory"])
            )

            # Get circuit metadata if it exists.
            if circ.metadata is not None:
                circ_metadata = circ.metadata
            else:
                circ_metadata = {}
            header = QobjExperimentHeader(metadata=circ_metadata)

            ## Create ExperimentResult
            result_data = {}
            # Add optional outputs depending on return_X options.

            # return_post_meas_state
            if kwargs.get(
                "return_post_meas_state", self.options.return_post_meas_state
            ):
                result_data["post_meas_state"] = data["post_meas_state"]

            # return_meas_state
            if kwargs.get("return_post_meas_state", self.options.return_meas_state):
                result_data["meas_state"] = data["meas_state"]

            # return_memory_labelled
            if kwargs.get(
                "return_memory_labelled", self.options.return_memory_labelled
            ):
                result_data["memory_labelled"] = data["memory_labelled"]

            # return_channel
            if kwargs.get("return_channel", self.options.return_channel):
                result_data["channel"] = data["channel"]

            # return_trans_mat
            if kwargs.get("return_trans_mat", self.options.return_trans_mat):
                result_data["trans_mat"] = data["transition_matrix"]
            ##

            ## Create experiment result instance
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
        ##

        ## Create RestlessJob
        job_id = str(uuid.uuid4())
        # cum_trans_mats is a result for the entire job and not an individual circuit, so we add it
        # to the restless job and not an experiment result.
        job_kwargs = {
            "job_id": job_id,
            "cum_trans_mats": cum_trans_mats
            if self.options.compute_cumulative_trans_mats
            else None,
        }
        job = RestlessJob(
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

        return job

    def configuration(self) -> RestlessBackendConfiguration:
        """Return the backend configuration."""
        return RestlessBackendConfiguration()

    def properties(self) -> RestlessBackendProperties:
        """Return the backend properties."""
        return RestlessBackendProperties()
