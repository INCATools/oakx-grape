"""GrapeImplementation test."""
import unittest

from oaklib.datamodels.vocabulary import IS_A
from oaklib.selector import get_implementation_from_shorthand

from tests import MORPHOLOGY, SHAPE


class TestGrapeImplementation(unittest.TestCase):
    """tests grape plugin."""

    def setUp(self) -> None:
        """Set up implementation."""
        self.oi = get_implementation_from_shorthand("grape:sqlite:obo:pato")
        self.predicates = ["rdfs:subClassOf"]
        self.bfo_subject = "BFO:0000006"
        self.bfo_object = "BFO:0000018"

    def test_entities(self):
        """Test basic functionality."""
        curies = list(self.oi.entities())
        self.assertIn(SHAPE, curies)
        self.assertIn(MORPHOLOGY, curies)

    # @unittest.skip("https://github.com/AnacletoLAB/ensmallen/issues/175")
    def test_labels(self):
        """Basic tests."""
        self.assertEqual("shape", self.oi.label(SHAPE))

    def test_edges(self):
        """
        Test retrieval of edges..

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

    # @unittest.skip("Need ensmallen help")
    def test_ancestors(self):
        """Test ability to traverse from a node to ancestors."""
        ancs = list(self.oi.ancestors(SHAPE))
        for a in ancs:
            print(f"{a}")

    # TODO: write simple test to verify one label (in the OAK sense, e.g. a description of a class or
    # something) - this will exercise the code to keep track of node properties, which will require
    # harmonizing the OAK and Grape representations of the ontology

    # wrap oak representation of graph
    # make tsv from this
    # instantiate grape using from_csv

    def test_pairwise_similarity(self):
        """Verify that pairwise similarity returns expected results."""
        oi = get_implementation_from_shorthand("grape:sqlite:obo:bfo")
        tp = oi.pairwise_similarity(self.bfo_subject, self.bfo_object)
        score = tp.ancestor_information_content
        self.assertGreaterEqual(len(tp), 7)
        self.assertGreater(score, 1.65)

        # And with predicates.
        tp = oi.pairwise_similarity(self.bfo_subject, self.bfo_object, self.predicates)
        score = tp.ancestor_information_content
        self.assertGreaterEqual(len(tp), 7)
        self.assertGreater(score, 1.65)

    def test_termset_pairwise_similarity(self):
        """Verify that termset similarity returns expected results."""
        oi = get_implementation_from_shorthand("grape:sqlite:obo:bfo")
        tp = oi.termset_pairwise_similarity([self.bfo_subject], [self.bfo_object])
        score = tp.subject_best_matches[self.bfo_subject].score
        self.assertGreaterEqual(len(tp), 7)
        self.assertGreater(score, 1.65)

        # And with predicates.
        tp = oi.termset_pairwise_similarity([self.bfo_subject], [self.bfo_object], self.predicates)
        score = tp.subject_best_matches[self.bfo_subject].score
        self.assertGreaterEqual(len(tp), 7)
        self.assertGreater(score, 1.65)

    def test_all_by_all_pairwise_similarity(self):
        """Verify that all by all pairwise similarity returns expected results."""
        oi = get_implementation_from_shorthand("grape:sqlite:obo:bfo")
        entities = list(oi.entities())
        tps = oi.all_by_all_pairwise_similarity(entities, entities)
        self.assertEqual(540, len(list(tps)))

        # And with predicates.
        tps = oi.all_by_all_pairwise_similarity(entities, entities, self.predicates)
        self.assertEqual(540, len(list(tps)))
