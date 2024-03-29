# Contributing

**We appreciate all kinds of help, so thank you!**

## Contributing to the restless simulator

Specific details for contributing to this project are outlined below.

### Reporting Bugs and Requesting Features

We encourage users to use GitHub Issues for reporting issues and requesting features.

### Ask/Answer Questions and Discuss the Restless Simulator

We encourage users to use GitHub Discussions for engaging with researchers, developers, and other 
users regarding this restless simulator and the provided examples.

### Project Code Style

Code in this repository should conform to PEP8 standards. Style/lint checks are run to 
validate this. Line length must be limited to no more than 88 characters.

### Pull Request Checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the _code style_ of this project and successfully
   passes the _unit tests_. This prototype uses [Pylint](https://www.pylint.org) and
   [PEP8](https://www.python.org/dev/peps/pep-0008) style guidelines.

   You can run
   ```shell script
   tox -elint
   ```
   from the root of the repository clone for lint conformance checks.
