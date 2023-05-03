# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
============================================================
Utilities module (:mod:`restless_simulator.utils`)
============================================================

.. currentmodule:: restless_simulator.utils

Restless Post-Processing Functions
==================================

.. autosummary::
    :toctree: ../stubs/

    extract_memory
    memory_to_probabilities
    restless_memory_to_memory

"""
import numpy as np

from restless_simulator.simulator import RestlessJob


def extract_memory(in_job: RestlessJob) -> np.ndarray:
    """Extract memory from ``in_job``, returned as a 2D NumPy array of single-shot outcome
    labels.

    Args:
        in_job: A :class:`RestlessJob` instance.

    Returns:
        A 2D NumPy array of shape ``(N_CIRCUITS, n_shots)`` containing the results of the job.
    """
    results = in_job.result().results
    # Convert single-shot results from hexadecimal strings to integers. Expected values are 0, 1,
    # and 2.
    memory = np.array(
        [list(map(lambda x: str(int(x, 16)), res.data.memory)) for res in results]
    )

    return memory


def restless_memory_to_memory(memory: np.ndarray) -> np.ndarray:
    """Convert restless memory measurements into standard measurements.

    This function applies restless post-processing to memory outcomes. It assumes that ``memory`` is
    a 2D array of shape ``(N_CIRCUITS, n_shots)`` where the circuits and shots are time-ordered. It
    also assumes that the entries are either of the strings ``"0"`` and ``"1"``.

    Args:
        memory: A 2D NumPy array of restless single-shot measurements.

    Returns:
       A 2D array of the same shape as ``memory``, but with the post-processed standard measurement
       outcomes based on restless circuit execution.
    """
    initial_shape = memory.shape
    ## Reshape memory into time-ordered shots, i.e.,
    # shots[0] -> circuit 0 shot 0
    # shots[1] -> circuit 1 shot 0
    # ...
    # shots[N_CIRCUITS] -> circuit 0 shot 1
    # ...
    memory = memory.reshape((-1), order="F").astype(int)

    ## Add input ground-state and compute XOR difference to reconstruct standard measurement
    ## outcomes.

    # input states, assuming initial ground-state
    input_memory = np.roll(memory, 1)
    input_memory[0] = 0

    # Return XOR between measurements and input states.
    processed_memory = np.logical_xor(memory, input_memory).astype(int).astype(str)

    # Return processed memory as original shape
    return np.reshape(processed_memory, initial_shape, order="F")


def memory_to_probabilities(memory: np.ndarray, outcome="0") -> np.ndarray:
    """Converts a memory array into a list of reconstructed probabilities to measure ``outcome``.

    This is done by dividing the number of occurrences $N_i$ of the outcome ``"i"`` in the memory
    array by the number of shots (i.e., ``memory.shape[1]``). This process is done along the second
    axis of ``memory`` so that there are individual probabilities per circuit.

    Args:
        memory: A 2D array of measurement outcomes ``"0"`` and ``"1"``, with shape ``(N_CIRCUITS,
            n_shots)``.
        outcome: The outcome label for the probabilities, either ``"0"`` or ``"1"``. Defaults to
            "0".

    Returns:
        A 1D array of reconstructed probabilities to measure ``outcome`` for each circuit.
    """
    return np.sum(memory == outcome, axis=1) / memory.shape[1]
