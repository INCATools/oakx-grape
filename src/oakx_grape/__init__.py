"""oakx_grape package."""
from importlib import metadata

from oakx_grape.grape_implementation import GrapeImplementation

__version__ = metadata.version(__name__)

schemes = {
    'grape': GrapeImplementation
}
