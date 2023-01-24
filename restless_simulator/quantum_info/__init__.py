"""
============================================================
Quantum Info module (:mod:`restless_simulator.quantum_info`)
============================================================

.. currentmodule:: restless_simulator.quantum_info

This module contains code for qutrit quantum information processing, such as defining qutrit
operators.

============
Base classes
============

.. autosummary::
    :toctree: ../stubs/

    Operator

=============================
Qutrit Quantum Error Channels
=============================

.. autosummary::
    :toctree: ../stubs/

    amplitude_damping_channel
    depolarizing_channel
"""

from .channels import amplitude_damping_channel, depolarizing_channel
from .operator import Operator
