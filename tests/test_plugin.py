"""GrapeImplementation test."""
import unittest

from oaklib.implementations import get_implementation_resolver
from oaklib.selector import get_resource_from_shorthand

from tests import TEST_OWL


class TestGrapeImplementation(unittest.TestCase):
    """Test GrapeImplementation plugin discovery."""

    def test_plugin(self):
        """Test plugins is discovered."""
        # This needs to be imported here to avoid circular imports
        from oakx_grape.grape_implementation import GrapeImplementation

        implementation_resolver = get_implementation_resolver()
        resolved = implementation_resolver.lookup("grape")
        self.assertEqual(resolved, GrapeImplementation)
        slug = f"grape:{TEST_OWL}"
        r = get_resource_from_shorthand(slug)
        self.assertEqual(r.implementation_class, GrapeImplementation)
