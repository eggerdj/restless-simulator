"""Restless Simulator."""

from importlib_metadata import version as metadata_version, PackageNotFoundError

try:
    __version__ = metadata_version("restless_simulator")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
