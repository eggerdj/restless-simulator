![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-informational)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%200.40.0-6133BD)](https://github.com/Qiskit/qiskit)
[![License](https://img.shields.io/github/license/qiskit-community/quantum-prototype-template?label=License)](https://github.com/eggerdj/restless-simulator/blob/main/LICENSE.txt)
[![Code style: Black](https://img.shields.io/badge/Code%20style-Black-000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/qiskit-community/quantum-prototype-template/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/eggerdj/restless-simulator/actions/workflows/test_latest_versions.yml)
[![Coverage](https://coveralls.io/repos/github/qiskit-community/quantum-prototype-template/badge.svg?branch=main)](https://coveralls.io/github/qiskit-community/quantum-prototype-template?branch=main)

# Restless simulator

This repository is a simulator to simulate restless circuit execution with qutrits [1].
In restless circuit execution, the qubits are not reset in between measurements [2, 3].
Leakage may therefore cause a build-up of population in states outside the computational
sub-space.
The simulator in this package helps explore these issues.
Importantly, the restless simulator computes the dynamics on the full matrices corresponding
to the quantum circuits.
The size of these matrices scales exponentially with the number of transmons in the circuit. 
This is fine since the restless simulator is intended to simulate characterization and calibration 
experiments which involve only a few transmons.
An example of the type of research that this repository enables is shown in Ref. [1].

### Table of Contents

##### For Users

1.  [About the Project](docs/project_overview.md)
2.  [Beginner's Guide](docs/beginners_guide.md)
3.  [Installation](INSTALL.md)
4.  [Tutorials](docs/tutorials/)
5.  [How-Tos](docs/how_tos/)
6.  [Restless Simulator Glossary](docs/file-map-and-description.md)
7.  [How to Give Feedback](#how-to-give-feedback)
8.  [Contribution Guidelines](#contribution-guidelines)
9. [References and Acknowledgements](#references-and-acknowledgements)
10. [License](#license)

##### For Developers/Contributors

1. [Contribution Guide](CONTRIBUTING.md)


----------------------------------------------------------------------------------------------------

### How to Give Feedback

We encourage your feedback! You can share your thoughts with us by:
- [Opening an issue](https://github.com/eggerdj/restless-simulator/issues) in the repository


----------------------------------------------------------------------------------------------------

### Contribution Guidelines

For information on how to contribute to this project, please take a look at our [contribution guidelines](CONTRIBUTING.md).


----------------------------------------------------------------------------------------------------

## References and Acknowledgements
[1] Conrad J. Haupt, Daniel J. Egger, Leakage in restless quantum gate calibration,
arxiv:2304.09297 (2023). https://arxiv.org/abs/2304.09297

[2] Caroline Tornow, Naoki Kanazawa, William E. Shanks, Daniel J. Egger,
Minimum quantum run-time characterization and calibration via restless
measurements with dynamic repetition rates, Physics Review Applied **17**,
064061 (2022). https://arxiv.org/abs/2202.06981

[3] Max Werninghaus, Daniel J. Egger, Stefan Filipp, High-speed calibration and
characterization of superconducting quantum processors without qubit reset,
PRX Quantum 2, 020324 (2021). https://arxiv.org/abs/2010.06576


----------------------------------------------------------------------------------------------------

### License
[Apache License 2.0](LICENSE.txt)
