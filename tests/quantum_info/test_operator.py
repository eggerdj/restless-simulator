"""Tests for qutrit Operator class."""
from typing import Tuple
import ddt
import numpy as np
from unittest import TestCase

from qiskit.exceptions import QiskitError

from restless_simulator.quantum_info import Operator

@ddt.ddt
class TestOperator(TestCase):
    """Tests qutrit operator."""

    @ddt.data(*range(9))
    def test_individual_barg_operators(self, barg_index:int):
        """Tests validity of individual Barg class."""
        label = f"B{barg_index}"
        actual_op = Operator.from_label(label)
        actual_mat = np.array(actual_op.data)

        # Construct expected operator matrix
        def int_to_ternary(value:int) -> Tuple[int, int]:
            """Converts ``value`` into ternary trits.

            Assumes ``value`` is a two-trit number.

            .. math::

                X = i\\times{}3^1 + j\\times{}3^0

            Args:
                value: The base-10 value to convert into ternary.

            Returns:
                The ternary bits ``i`` and ``j``.
            """
            if value < 0 or value > 8 or not isinstance(value,int):
                raise QiskitError(f"Value {value} is not a two-trit number!")
            i = int(value/3)
            j = value % 3
            return i,j
        
        barg_x = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)
        barg_w = np.exp(2j * np.pi / 3)
        barg_z = np.diag([1, barg_w, barg_w**2])
        i,j = int_to_ternary(barg_index)
        expected_mat = np.linalg.matrix_power(barg_x,i) @ np.linalg.matrix_power(barg_z,j)

        self.assertTrue(np.array_equal(expected_mat,actual_mat),f"Actual Barg operator matrix is not as expected. Expected\n{expected_mat}\n, instead got\n{actual_mat}.")
