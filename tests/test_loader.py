"""GrapeImplementation test."""
import unittest

from oaklib.selector import get_implementation_from_shorthand
from oakx_grape.loader import load_graph_from_adapter

from tests import TEST_DB


class TestLoader(unittest.TestCase):

    def test_loader(self):
        core_oi = get_implementation_from_shorthand(f"sqlite:{TEST_DB}")
        oi = load_graph_from_adapter(core_oi)

