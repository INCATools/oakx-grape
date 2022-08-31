"""oakx_grape package."""
from importlib import metadata

__version__ = metadata.version(__name__)

from oakx_grape import GrapeImplementation

schemes = {
    'grape': GrapeImplementation
}
