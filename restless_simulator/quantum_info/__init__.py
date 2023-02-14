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
Quantum Info module (:mod:`restless_simulator.quantum_info`)
============================================================

.. currentmodule:: restless_simulator.quantum_info

This module contains code for qutrit quantum information processing, such as defining qutrit
operators.

Base classes
============

.. autosummary::
    :toctree: ../stubs/

    Operator

Qutrit Quantum Error Channels
=============================

.. autosummary::
    :toctree: ../stubs/

    amplitude_damping_channel
    depolarizing_channel
"""

from .channels import amplitude_damping_channel, depolarizing_channel
from .operator import Operator
from .converters import qudit_circuit_to_super_op
