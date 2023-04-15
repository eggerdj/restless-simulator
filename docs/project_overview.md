# Project Overview

## Background

Restless circuit execution is a method to execute characterization and calibration
quantum circuits without resetting the qubits after each measurement.
The measurement must be projective, and the outcome of one circuit is the initial
state to the next quantum circuit that the hardware executes.
The qubits therefore sometimes begin the `|1>` state which is
accounted for by post-processing.
This allows faster circuit execution and speeds-up tasks such as gate calibration
and randomized benchmarking.

However, most quantum architectures embed a qubit in a larger Hilbert space.
This is the case for superconducting transmon qubits.
For transmon qubits improperly calibrated pulses may cause population to exit
the qubit sub-space, spanned by `{|0>, |1>}`, and enter into
higher excited states, typically the `|2>` state.
This effect is known as leakage.
Since restless circuit execution does not reset the qubits we may wonder
how leakage builds-up population outside the qubit sub-space and whether
this build-up is detrimental to the calibration and characterization tasks.

## Solution Explanation

The restless simulator in this package allows researchers to explore leakage in 
restless measurements.
The restless simulator builds transition matrices for the circuits that it simulates.
The instructions in these circuits embed qutrit operations that act on the first 
three levels of the transmons to simulate leakage.
The simulator computes these transition matrices from the quantum channels 
describing the circuits.
This implies that the simulator supports both unitary and non-unitary dynamics.
Furthermore, the simulator does not support large quantum circuits due to the
exponential growth of the Hilbert space.
Nevertheless, this is sufficient for characterization and calibration tasks which
typically only involve a few qubits at a time.
