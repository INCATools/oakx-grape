import csv
import inspect
import tempfile
from dataclasses import dataclass
from typing import Callable, ClassVar, Iterable, Iterator, List, Optional, Tuple, Union, Mapping

from ensmallen import Graph
from oaklib import BasicOntologyInterface, OntologyResource
from oaklib.datamodels.similarity import TermPairwiseSimilarity
from oaklib.datamodels.vocabulary import IS_A
from oaklib.implementations import SqlImplementation
from oaklib.interfaces.basic_ontology_interface import RELATIONSHIP_MAP
from oaklib.interfaces.semsim_interface import SemanticSimilarityInterface
from oaklib.types import CURIE, PRED_CURIE

# Mappings between biolink predicates and RO/OWL/RDF
# This won't be necessary once we load the ensmallen graph directly
# TODO: move this to OAK-central
from oaklib.utilities.basic_utils import pairs_as_dict
from oakx_grape.loader import load_graph_from_adapter

PREDICATE_MAP = {"biolink:subclass_of": IS_A}

GRAPH_PAIR = Tuple[Graph, Graph]


def get_graph_function_by_name(name: str, module="kgobo") -> Callable:
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
    the main graph. In this graph, ensmallen "neighbors" corresponds
    """
    transposed_graph: Graph = None
    """
    the main graph, with inverted directions. ensmallen does not handle child traversal,
    so we maintain two graphs
    """

    uses_biolink: bool = None

    _cached_graphs_by_predicates: Mapping[Tuple, GRAPH_PAIR] = None

    wrapped_adapter: BasicOntologyInterface = None
    """An OAK implementation that takes care of everything that ensmallen cannot handle"""

    delegated_methods: ClassVar[List[str]] = [
        "label",
        "labels",
        "curie_to_uri",
        "uri_to_curie",
        "ontologies",
        "obsoletes",
        "basic_search",
    ]
    """all methods that should be delegated to wrapped_adapter"""

    def __post_init__(self):
        slug = self.resource.slug
        # we delegate to two different implementations
        # 1. an ensmallen_graph
        # 2. a sqlite implementation
        # For now we load both from different sources
        # (1: kgobo; 2: bbop-sqlite).
        # This is a dumb temporary measure for testing.
        # In future we will first use the wrapped adapter to to get the graph,
        # and communicate it to graph via something like dumping files then loading
        if slug.startswith("kgobo:"):
            slug = slug.replace("kgobo:", "")
            self.uses_biolink = True
            f = get_graph_function_by_name(slug.upper())
            if self.wrapped_adapter is None:
                self.wrapped_adapter = SqlImplementation(OntologyResource(f"obo:{slug.lower()}"))
            self.graph = f(directed=True)
            self.transposed_graph = self.graph.to_transposed()
        else:
            from oaklib.selector import get_implementation_from_shorthand
            inner_oi = get_implementation_from_shorthand(slug)
            print(f"OI={inner_oi}")
            self.wrapped_adapter = inner_oi
            self.graph = load_graph_from_adapter(inner_oi)
            #self.transposed_graph = load_graph_from_adapter(inner_oi, transpose=True)
            self.transposed_graph = self.graph.to_transposed()
        # delegation magic
        methods = dict(inspect.getmembers(self.wrapped_adapter))
        for m in self.delegated_methods:
            setattr(GrapeImplementation, m, methods[m])

    def map_biolink_predicate(self, predicate: PRED_CURIE) -> PRED_CURIE:
        """
        Maps from biolink (use in KGX) to RO/OWL.

        Note this is only necessary for graphs from kgx obo

        :param predicate:
        :return:
        """
        if self.uses_biolink:
            return PREDICATE_MAP.get(predicate, predicate)
        else:
            return predicate

    def _load_graph_from_adapter(self, oi: BasicOntologyInterface):
        self.graph = load_graph_from_adapter(oi)
        self.transposed_graph = g.to_transposed()

    def _graph_pair_by_predicates(self, predicates: List[PRED_CURIE] = None):
        # note the size of the cache will grow with each distinct combination of
        # predicates used
        if self._cached_graphs_by_predicates is None:
            self._cached_graphs_by_predicates = {}
        if predicates is None:
            predicates = []
        tp = tuple(sorted(list(predicates)))
        if tp in self._cached_graphs_by_predicates:
            return self._cached_graphs_by_predicates[tp]
        filtered_graph = self.graph.filter_from_names(
            edge_type_names_to_keep=predicates,
        )
        filtered_transposed_graph = self.transposed_graph.filter_from_names(
            edge_type_names_to_keep=predicates,
        )
        self._cached_graphs_by_predicates[tp] = (filtered_graph, filtered_transposed_graph)
        return filtered_graph, filtered_transposed_graph

    def entities(self, filter_obsoletes=True, owl_type=None) -> Iterable[CURIE]:
        g = self.graph
        for n_id in g.get_node_ids():
            yield g.get_node_name_from_node_id(n_id)

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
            if predicates and pred not in predicates:
                continue
            yield pred, obj

    def outgoing_relationship_map(self, *args, **kwargs) -> RELATIONSHIP_MAP:
        return pairs_as_dict(self.outgoing_relationships(*args, **kwargs))

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
            if predicates and pred not in predicates:
                continue
            yield pred, subj

    def ancestors(
        self,
        start_curies: Union[CURIE, List[CURIE]],
        predicates: List[PRED_CURIE] = None,
        reflexive=True,
    ) -> Iterable[CURIE]:
        g = self.graph
        anc_ids = {}
        if not isinstance(start_curies, list):
            start_curies = [start_curies]
        for curie in start_curies:
            curie_id = g.get_node_id_from_node_name(curie)
            anc_ids.update(g.get_successors_from_node_id(curie_id))
        for anc_id in anc_ids:
            yield g.get_node_name_from_node_id(anc_id)

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
