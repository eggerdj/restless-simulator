# Restless Simulator Beginner's Guide

This beginner's guide shows how to install the restless-simulator and points you to the 
short tutorials that should get you started.
In these tutorials you will learn how to simulate quantum circuits with restless execution, i.e.,
without qubit reset, and with leakage out of the computational sub-space.

## Installation
The project can be installed by forking and cloning the code from Github.
Once cloned to your local machine, you can install the code with `pip install .`.

## Usage
Standard circuits run on qubits can be converted to qutrit circuits with the utilities in
the `quantum_info` module.
The restless simulator in `restless_simulator.simulator` can then simulate the time-ordered
shots that restless circuit execution produces.

## Example Problem
A small example problem is in the tutorial 
[Introduction to the simulator](docs/tutorials/1_introduction_to_simulator.ipynb)
where we show how to simulate a sequence of rotations around the `X` with and without
restless data processing.
In this tutorial you learn how to use the `QutritRestlessSimulator`.
The second tutorial [Simulating qutrit circuits](docs/tutorials/2_simulating_qutrit_circuits.ipynb)
shows what happens when the quantum gates have leakage in them.
Here, you learn how to use the `QutritUnitaryGate` class and how to add leakage to them.

## Conclusion

In summary, this package provides the tools needed to simulate restless circuit execution with leakage.
