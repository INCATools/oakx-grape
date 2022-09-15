"""GrapeImplementation test."""
from tests import TEST_OWL
from tests import NUCLEUS
from oaklib.selector import get_resource_from_shorthand
from oaklib.implementations import implementation_resolver

import unittest

class TestGrapeImplementation(unittest.TestCase):
    """Test GrapeImplementation."""
        
    def test_plugin(self):
        """tests plugins are discovered"""
        # This needs to be imported here to avoid circular imports
        from oakx_grape.grape_implementation import GrapeImplementation
        
        resolved = implementation_resolver.lookup("grape")
        self.assertEqual(resolved, GrapeImplementation)
        slug = f"grape:{TEST_OWL}"
        r = get_resource_from_shorthand(slug)
        self.assertEqual(r.implementation_class, GrapeImplementation)

    # def test_all(self):
    #     """
    #     Test basic functionality
    #     """
    #     curies = list(self.oi.all_entity_curies())
    #     self.assertIn(NUCLEUS, curies)

