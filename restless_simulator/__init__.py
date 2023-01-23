"""Template project."""

from importlib_metadata import version as metadata_version, PackageNotFoundError

from .circuit import TemplateClass


try:
    __version__ = metadata_version("restless_simulator")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
