"""GrapeImplementation test."""
from oakx_grape.grape_implementation import GrapeImplementation
from tests import TEST_OWL
from tests import NUCLEUS
from oaklib.selector import get_resource_from_shorthand, discovered_plugins, get_implementation_from_shorthand

import unittest

class TestGrapeImplementation(unittest.TestCase):
    """Test GrapeImplementation."""

    def setUp(self) -> None:
        self.oi = get_implementation_from_shorthand(f"grape:{TEST_OWL}")
        
    def test_plugin(self):
        """tests plugins are discovered"""
        plugins = discovered_plugins
        self.assertIn("oakx_grape", plugins)
        slug = f"grape:{TEST_OWL}"
        r = get_resource_from_shorthand(slug)
        self.assertEqual(r.implementation_class, GrapeImplementation)

    def test_all(self):
        """
        Test basic functionality
        """
        curies = list(self.oi.all_entity_curies())
        self.assertIn(NUCLEUS, curies)

    # TODO: write simple test to verify one label (in the OAK sense, e.g. a description of a class or
    # something) - this will exercise the code to keep track of node properties, which will require
    # harmonizing the OAK and Grape representations of the ontology

    # wrap oak representation of graph
    # make tsv from this
    # instantiate grape using from_csv
