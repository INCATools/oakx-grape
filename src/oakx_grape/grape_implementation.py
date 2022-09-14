from dataclasses import dataclass
from typing import Iterable, Optional, List, Iterator, Tuple, Callable

from oaklib import BasicOntologyInterface
import grape
from ensmallen import Graph
from oaklib.datamodels.similarity import TermPairwiseSimilarity
from oaklib.datamodels.vocabulary import IS_A
from oaklib.interfaces.semsim_interface import SemanticSimilarityInterface
from oaklib.types import CURIE, PRED_CURIE

# Mappings between biolink predicates and RO/OWL/RDF
# TODO: move this to OAK-central
PREDICATE_MAP = {
    "biolink:subclass_of": IS_A
}

def get_graph_function_by_name(name: str, module = "kgobo") -> Callable:
    """Dynamically import a Grape class based on its reference.

    :param name: e.g. PATO
    :param module: e.g. kgobo (default)
    :return: A function that can be called to to create a graph
    """
    mod = __import__(f"grape.datasets.{module}", fromlist=[name])
    return getattr(mod, name)


@dataclass
class GrapeImplementation(SemanticSimilarityInterface):
    """
    An experimental wrapper for Grape/Ensmallen
    """
    graph: Graph = None
    """
    the main graph. In this graph, ensmalled "neighbors" corresponds
    """
    transposed_graph: Graph = None

    def __post_init__(self):
        f = get_graph_function_by_name(self.resource.slug)
        self.graph = f(directed=True)
        self.transposed_graph = self.graph.to_transposed()

    def _get_grape_csv_location(self):
        #self.graph.introspect_cache_path
        pass

    def map_biolink_predicate(self, predicate: PRED_CURIE) -> PRED_CURIE:
        """
        Maps from biolink to RO/OWL
        :param predicate:
        :return:
        """
        return PREDICATE_MAP.get(predicate, predicate)

    def entities(self, filter_obsoletes=True, owl_type=None) -> Iterable[CURIE]:
        g = self.graph
        for n_id in g.get_node_ids():
            yield g.get_node_name_from_node_id(n_id)

    #def curies_by_label(self, label: str) -> List[CURIE]:
    #    g = self.graph
    #    g.get_node_id_from_node_name(label)

    def label(self, curie: CURIE) -> Optional[str]:
        pass

    def outgoing_relationships(
            self, curie: CURIE, predicates: List[PRED_CURIE] = None
    ) -> Iterator[Tuple[PRED_CURIE, CURIE]]:
        g = self.graph
        curie_id = g.get_node_id_from_node_name(curie)
        for object_id in g.get_neighbour_node_ids_from_node_id(curie_id):
            obj = g.get_node_name_from_node_id(object_id)
            edge_id = g.get_edge_id_from_node_ids(curie_id, object_id)
            pred = g.get_edge_type_name_from_edge_id(edge_id)
            pred = self.map_biolink_predicate(pred)
            yield pred, obj

    def incoming_relationships(
            self, curie: CURIE, predicates: List[PRED_CURIE] = None
    ) -> Iterator[Tuple[PRED_CURIE, CURIE]]:
        g = self.transposed_graph
        curie_id = g.get_node_id_from_node_name(curie)
        for subject_id in g.get_neighbour_node_ids_from_node_id(curie_id):
            subj = g.get_node_name_from_node_id(subject_id)
            edge_id = g.get_edge_id_from_node_ids(curie_id, subject_id)
            pred = g.get_edge_type_name_from_edge_id(edge_id)
            pred = self.map_biolink_predicate(pred)
            yield pred, subj

    # -- SemSim methods --

    def pairwise_similarity(
            self,
            subject: CURIE,
            object: CURIE,
            predicates: List[PRED_CURIE] = None,
            subject_ancestors: List[CURIE] = None,
            object_ancestors: List[CURIE] = None,
    ) -> TermPairwiseSimilarity:
        if predicates:
            raise ValueError(f"For now can only use hardcoded ensmallen predicates")
        raise NotImplementedError

