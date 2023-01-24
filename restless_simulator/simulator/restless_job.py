# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Job class for the QutritRestlessSimulator"""
import uuid

from qiskit.providers import Backend, JobStatus
from qiskit.providers.job import JobV1
from qiskit.result import Result


class RestlessJob(JobV1):
    """Restless Simulator Job Class."""

    _async = False

    def __init__(self, backend: Backend, result: Result, job_id: str = None):
        """Create a new RestlessJob instance.

        Args:
            backend: The backend used to create the job.
            result: The result object for the simulation.
            job_id: Optional job id. If none, a new job id is created. Defaults to None.
        """
        if job_id is None:
            job_id = str(uuid.uuid4())
        super().__init__(backend, job_id)
        self._result = result

    def submit(self):
        """Submit the job to the simulator.

        Does nothing.
        """
        pass

    def result(self) -> Result:
        """Get job result.

        Returns:
            Result: Result object
        """
        return self._result

    def status(self) -> JobStatus:
        """Gets the status of the simulator job.

        As the simulator does not run asynchronously, the job result is always done. Until the simulator supports 

        Returns:
            JobStatus: The current JobStatus.
        """
        return JobStatus.DONE

    def backend(self) -> Backend:
        """Return the simulator instance used for this job."""
        return self._backend
