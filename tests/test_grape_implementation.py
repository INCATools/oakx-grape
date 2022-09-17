"""GrapeImplementation test."""
import unittest

from oaklib.datamodels.vocabulary import IS_A
from oaklib.selector import get_implementation_from_shorthand

from tests import MORPHOLOGY, SHAPE


class TestGrapeImplementation2(unittest.TestCase):
    """TEMP until https://github.com/INCATools/ontology-access-kit/issues/250 fixes."""

    def setUp(self) -> None:
        self.oi = get_implementation_from_shorthand("grape:sqlite:obo:pato")
        #self.oi = get_implementation_from_shorthand("grape:kgobo:pato")
        #self.oi = get_implementation_from_shorthand("grape:pato")
        # self.oi = GrapeImplementation(OntologyResource("PATO"))

    def test_entities(self):
        """
        Test basic functionality
        """
        curies = list(self.oi.entities())
        self.assertIn(SHAPE, curies)
        self.assertIn(MORPHOLOGY, curies)

    # @unittest.skip("https://github.com/AnacletoLAB/ensmallen/issues/175")
    def test_labels(self):
        """
        Test basic functionality
        """
        self.assertEqual("shape", self.oi.label(SHAPE))

    def test_edges(self):
        """
        Test basic functionality?

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

    @unittest.skip("Need ensmallen help")
    def test_ancestors(self):
        """
        Test ability to traverse from a node to ancestors
        """
        ancs = list(self.oi.ancestors(SHAPE))
        for a in ancs:
            print(f"{a}")


