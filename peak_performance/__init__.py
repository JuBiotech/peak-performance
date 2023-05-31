import importlib.metadata
from . import models
from . import plots

__version__ = importlib.metadata.version(__package__ or __name__)
"""Package version when the install command ran."""
