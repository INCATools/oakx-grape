"""GrapeImplementation test."""
from oaklib import OntologyResource
from oaklib.datamodels.vocabulary import IS_A

from oakx_grape.grape_implementation import GrapeImplementation
from tests import TEST_OWL, MORPHOLOGY, SHAPE
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
        self.assertEqual("nucleus", self.oi.label(NUCLEUS))

    def test_edges(self):
        """
        Test basic functionality

        We use a basic edge from PATO

         - SHAPE subClassOf MORPHOLOGY
        """
        rels = list(self.oi.outgoing_relationships(SHAPE))
        for pred, subj in rels:
            print(f"{pred} {subj}")
        self.assertIn((IS_A, MORPHOLOGY), rels)
        rels = list(self.oi.incoming_relationships(MORPHOLOGY))
        for pred, subj in rels:
            print(f"{pred} {subj}")
        self.assertIn((IS_A, SHAPE), rels)



