"""GrapeImplementation test."""
from oaklib import OntologyResource
from oaklib.datamodels.vocabulary import IS_A

from oakx_grape.grape_implementation import GrapeImplementation
from tests import MORPHOLOGY, SHAPE

import unittest

class TestGrapeImplementation2(unittest.TestCase):
    """TEMP until https://github.com/INCATools/ontology-access-kit/issues/250 fixes."""

    def setUp(self) -> None:
        self.oi = GrapeImplementation(OntologyResource("PATO"))

    def test_entities(self):
        """
        Test basic functionality
        """
        curies = list(self.oi.entity_curies())
        self.assertIn(SHAPE, curies)
        self.assertIn(MORPHOLOGY, curies)

    @unittest.skip("https://github.com/AnacletoLAB/ensmallen/issues/175")
    def test_labels(self):
        """
        Test basic functionality
        """
        self.assertEqual("shape", self.oi.label(SHAPE))

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



