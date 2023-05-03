Simulating Leakage in Restless Quantum Circuit Execution
========================================================

Here, we show how to perform a simulation of restless circuit execution
for the case where the underlying gates have leakage.
To do this you need to

* Add `QutritQuantumChannelOperation` or `QutritUnitaryGate` instructions to the quantum circuit,
  these classes are for non-unitary and unitary dynamics, respectively.

* To create these instructions you will need to create the matrices that define them. For unitary
  qutrit gates these are 3x3 unitary matrices.

* Finally you will simulate the circuit using the `QutritRestlessSimulator` class and post-process the
  measured shots with the utility function `restless_memory_to_memory`.