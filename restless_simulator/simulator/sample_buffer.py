"""Sample buffer for speeding up computation of measurement results from transition matrices."""
from collections import deque
from typing import List

import numpy as np


class SampleBuffer:
    """A sample buffer to store precomputed samples from a transition matrix.

    Instead of sampling from a transition matrix ``trans_mat`` for each shot, :class:`SampleBuffer`
    stores :math:`N` pre-computed outcomes for all possible input states. These outcomes are stored
    in buffers which refill when a certain minimum count is reached. Example usage of
    :class:`SampleBuffer` is given below.

    .. code-block::

        # Get transition matrix
        trans_mat = gen_trans_mat(...)

        n_shots = 1024
        samples = SampleBuffer(trans_mat, size=n_shots)

        # Get restless shots
        prev_state = 0  # Start in the ground-state.
        measurements = []
        for _ in range(n_shots):
            prev_state = samples.get_label(prev_state)
            measurements.append(prev_state)
    """

    def __init__(
        self, transition_matrix: np.ndarray, size: int = 1024, when_to_fill: int = 0
    ):
        """Creates a sample buffer for the given transition matrix.

        The buffers are automatically filled on construction.

        Args:
            transition_matrix: The transition matrix to be sampled.
            size: The number of samples to buffer. Defaults to 1024.
            when_to_fill: The number of samples left in the buffer when more samples should be
                generated.
        """
        self._trans_mat = transition_matrix
        self._size = size
        self._when_to_fill = when_to_fill
        self._out_ndims, self._in_ndims = transition_matrix.shape
        self._queues: List[deque] = [
            deque(maxlen=self._size) for _ in range(self._in_ndims)
        ]

        # Populate queues
        for i in range(self._in_ndims):
            self._fill_queue(i)

    @property
    def in_ndims(self) -> int:
        """The number of dimensions as the input to the sampled transition matrix."""
        return self._in_ndims

    @property
    def out_ndims(self) -> int:
        """The number of dimensions as the output from the sampled transition matrix."""
        return self._out_ndims

    def _fill_queue(self, in_val: int):
        """Fill the queue for input value ``in_val``.

        Args:
            in_val: The input value identifying the sample buffer to fill.
        """

        n_add = self._size - len(self._queues[in_val])
        if n_add <= 0:
            return

        # Generate additional samples
        additional = np.random.choice(
            range(self._out_ndims), size=n_add, p=self._trans_mat[:, in_val].flatten()
        )
        self._queues[in_val].extend(additional)

    def _queue_needs_filling(self, in_val: int) -> bool:
        """Whether the queue/buffer for the input value ``in_val`` needs to be refilled.

        Args:
            in_val: The input state index.

        Returns:
            If the queue for input state ``in_val`` needs to be filled; i.e., its size is less-than
            or equal to :attr:`_when_to_fill`.
        """
        return len(self._queues[in_val]) <= self._when_to_fill

    def get_label(self, state_index: int) -> int:
        """State label sampled from the transition matrix given the input state ``state_index``.

        Args:
            state_index: The input state index, as if an input vector was multiplied by the
                transition matrix.

        Returns:
            The output state index from the pre-computed samples.
        """
        if self._queue_needs_filling(state_index):
            self._fill_queue(state_index)
        return self._queues[state_index].pop()
