"""GrapeImplementation test."""
from oaklib import OntologyResource

from oakx_grape.grape_implementation import GrapeImplementation
from tests import TEST_OWL
from tests import NUCLEUS
from oaklib.selector import get_resource_from_shorthand, discovered_plugins, get_implementation_from_shorthand

import unittest

class TestGrapeImplementation2(unittest.TestCase):
    """TEMP until https://github.com/INCATools/ontology-access-kit/issues/250 fixes."""

    def setUp(self) -> None:
        self.oi = GrapeImplementation(OntologyResource("PATO"))

    def test_all(self):
        """
        Test basic functionality
        """
        curies = list(self.oi.all_entity_curies())
        self.assertIn(NUCLEUS, curies)

