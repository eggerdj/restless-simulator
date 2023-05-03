Simulating Restless Quantum Circuit Execution
=============================================

Here, we show how to simulate a quantum circuit in a restless fashion.
We will simulate ten shots of a deterministic quantum circuit made of
an `X` gate and a `SWAP` gate.
We chose a deterministic quantum circuit so that we can easily follow
through the order of the shots.

.. jupyeter-execute::

    from qiskit import QuantumCircuit

    circ = QuantumCircuit(2)
    circ.x(0)
    circ.swap(0, 1)
    circ.draw("mpl")

The restless simulator assumes that each wire in a circuit is measured.
There is therefore no need to include measurement instructions at the end of
each quantum circuit.
We now simulate the circuit above assuming that there is no reset of the qubits.

.. jupyeter-execute::

    from restless_simulator.simulator import QutritRestlessSimulator

    backend = QutritRestlessSimulator()
    result = backend.run(circ, shots=10).result()

    print(result.get_memory())

The memory displays ten outcomes that seem a priori unrelated.
There is however an order to the memory outcomes that are display.
The first shots yields `10` since qubit 0 is excited an then swapped
with qubit 1.
This means that the initial state for the next shot is `10`. The `X`
gate excites qubit 0 (which was measured in state `0` in the last shot).
Qubit 0 and 1 are now both in state `1` and so the next measured shot
produces the outcome `11`.
By continuing this reasoning the next outcome is `01`.
This short example shows how to create measurement outcomes from a restless
circuit execution method.
