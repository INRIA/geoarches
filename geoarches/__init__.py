"""Tools for training and evaluating geospatial machine-learning models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("geoarches")
except PackageNotFoundError:
    # The package may be imported directly from a source checkout.
    __version__ = "unknown"

__all__ = ["__version__"]
