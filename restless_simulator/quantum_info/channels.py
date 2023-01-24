# This code is a Qiskit project.

# (C) Copyright IBM 2023.

# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Qutrit Quantum Channels"""
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Kraus

from .operator import Operator


def amplitude_damping_channel(
    param10: float, param21: float, param20: float = 0
) -> Kraus:
    r"""Return a qutrit amplitude damping channel.

    The amplitude damping channel for a single qutrit in the operator-sum representation

    .. math::

        \epsilon(\rho) = \sum_{i=0}^3{K_i\rho{}K_i^\dagger}

    has four Kraus operators

    .. math::

        K_0 =
        \begin{bmatrix}
        1   & 0                 & 0                          \\
        0   & \sqrt{1 - p_{10}} & 0                          \\
        0   & 0                 & \sqrt{1 - p_{21} - p_{20}} \\
        \end{bmatrix},
        K_1 =
        \begin{bmatrix}
        0   & \sqrt{p_{10}} & 0 \\
        0   & 0             & 0 \\
        0   & 0             & 0 \\
        \end{bmatrix}

    .. math::

        K_2 =
        \begin{bmatrix}
        0   & 0     & 0             \\
        0   & 0     & \sqrt{p_{21}} \\
        0   & 0     & 0             \\
        \end{bmatrix},
        K_3 =
        \begin{bmatrix}
        0   & 0     & \sqrt{p_{20}} \\
        0   & 0     & 0             \\
        0   & 0     & 0             \\
        \end{bmatrix}

    where :math:`0\leq{}p_{ij}\leq{}1\ \forall{}i,j\in\{0,1,2\}` and :math:`0\leq{}p_{21} +
    p_{20}\leq{}1`. The amplitude damping parameters :math:`p_{ij}` define the decay rate from state
    `i` to state `j`. For the relaxation time :math:`T_1` of a qubit, the relaxation parameter
    :math:`p_{10}` is typically defined as :math:`p_{10} = 1 - exp(-\Delta{}t/T_1)` for a given
    gate-time :math:`\Delta{}t`.

    Args:
        param10: the :math:`\ket{1}-\ket{0}` damping parameter.
        param21: the :math:`\ket{2}-\ket{1}` damping parameter.
        param20: the :math:`\ket{2}-\ket{0}` damping parameter.

    Raises:
        QiskitError: if ``param10``, ``param21``, and ``param20`` are less than zero or too large.

    Returns:
        An amplitude damping channel in the Kraus representation.
    """

    # Verify parameter values
    _params_ndarray = np.asarray([param10, param21, param20])
    if any(_params_ndarray < 0) or any(_params_ndarray > 1):
        raise QiskitError(
            "Amplitude damping parameters must be between 0 and 1: param10="
            f"{param10}, param21={param21}, param20={param20}."
        )
    if param20 + param21 > 1:
        raise QiskitError(
            "Second excited state damping parameters must sum to less than 1: param20 + param21 = "
            f"{param20 + param21}"
        )

    k_0 = np.diag([1, np.sqrt(1 - param10), np.sqrt(1 - param20 - param21)]).astype(
        complex
    )
    k_1 = np.array([[0, np.sqrt(param10), 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
    k_2 = np.array([[0, 0, 0], [0, 0, np.sqrt(param21)], [0, 0, 0]], dtype=complex)
    k_3 = np.array([[0, 0, np.sqrt(param20)], [0, 0, 0], [0, 0, 0]], dtype=complex)

    # Create Kraus object for amplitude damping channel
    kraus = Kraus([k_0, k_1, k_2, k_3])
    return kraus


# This function is based on `qiskit_aer.noise.depolarizing_error`.
def depolarizing_channel(param: float) -> Kraus:
    r"""Return a qutrit depolarizing channel with depolarizing parameter ``param``.

    The depolarizing channel is modelled as

    .. math::

        \epsilon(\rho) = (1-p)\rho + p{}Tr[\rho]\frac{I}{3^n}

    for :math:`n` qutrits and probability :math:`p`. In the operator-sum/Kraus representation, this
    is given as

    .. math::

        \epsilon(\rho) = \sum_{i=0}^{N}{B_i\rho{}B_i^\dagger}

    where :math:`B_i` is the ith Barg matrix.

    References:
        [1] A. Barg, ‘A low-rate bound on the reliability of a quantum discrete memoryless channel’,
        IEEE Transactions on Information Theory, vol. 48, no. 12, pp. 3096–3100, Dec. 2002, doi:
        10.1109/TIT.2002.805080.

    Args:
        param: The depolarizing channel parameter.

    Raises:
        QiskitError: if ``param`` is less than zero or too large.

    Returns:
        A depolarizing channel in a Kraus representation.
    """
    # Get qutrit ops, first will be L0, which is the qutrit identity.
    qutrit_ops = [Operator.from_label(f"B{i}") for i in range(0, 9)]

    # Verify parameter is correct. This check is based on code from Qiskit-Aer, for the qubit
    # depolarizing error.
    num_qubits = 1
    num_terms = 9**num_qubits
    # Maximum depolarizing channel parameter, for a uniform Barg-operators channel: a qutrit
    # analogue of a uniform Pauli channel.
    max_param = num_terms / (num_terms - 1)
    if param < 0 or param > max_param:
        raise QiskitError(
            f"Depolarizing parameter must be between 0 and {max_param}, got {param:0.4e} "
            "instead."
        )

    # Calculate probabilities
    prob_identity = 1 - param / max_param
    prob_ops = param / num_terms
    probs = [prob_identity] + (num_terms - 1) * [prob_ops]

    # Create Kraus object for depolarizing channel
    kraus = Kraus([np.sqrt(p) * op for p, op in zip(probs, qutrit_ops)])
    return kraus
