import importlib.metadata

from . import models, pipeline, plots

__version__ = importlib.metadata.version(__package__ or __name__)
"""Package version when the install command ran."""

__all__ = (
    "__version__",
    "models",
    "pipeline",
    "plots",
)
