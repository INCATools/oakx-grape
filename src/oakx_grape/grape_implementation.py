"""Plugin for ensmallen/grape."""
import inspect
import logging
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, Iterable, Iterator, List, Optional, Tuple

from embiggen.edge_prediction.edge_prediction_ensmallen.perceptron import PerceptronEdgePrediction
from embiggen.embedders.ensmallen_embedders.first_order_line import FirstOrderLINEEnsmallen
from ensmallen import Graph
from oaklib import BasicOntologyInterface, OntologyResource
from oaklib.datamodels.similarity import TermPairwiseSimilarity
from oaklib.datamodels.vocabulary import IS_A
from oaklib.implementations import SqlImplementation
from oaklib.interfaces import SubsetterInterface
from oaklib.interfaces.basic_ontology_interface import RELATIONSHIP_MAP
from oaklib.interfaces.differ_interface import DifferInterface
from oaklib.interfaces.mapping_provider_interface import MappingProviderInterface
from oaklib.interfaces.metadata_interface import MetadataInterface
from oaklib.interfaces.obograph_interface import OboGraphInterface
from oaklib.interfaces.patcher_interface import PatcherInterface
from oaklib.interfaces.relation_graph_interface import RelationGraphInterface
from oaklib.interfaces.search_interface import SearchInterface
from oaklib.interfaces.semsim_interface import SemanticSimilarityInterface
from oaklib.interfaces.validator_interface import ValidatorInterface
from oaklib.types import CURIE, PRED_CURIE
from grape.similarities import DAGResnik

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
class GrapeImplementation(
    RelationGraphInterface,
    OboGraphInterface,
    ValidatorInterface,
    SearchInterface,
    SubsetterInterface,
    MappingProviderInterface,
    PatcherInterface,
    SemanticSimilarityInterface,
    MetadataInterface,
    DifferInterface,
):
    """
    An experimental wrapper for Grape/Ensmallen.

    This is intended primarily for semsim.
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

    uses_biolink: Optional[bool] = None

    _cached_graphs_by_predicates: Optional[Dict[Tuple, GRAPH_PAIR]] = None

    wrapped_adapter: BasicOntologyInterface = None
    """An OAK implementation that takes care of everything that ensmallen cannot handle"""

    delegated_methods: ClassVar[List[str]] = [
        BasicOntologyInterface.label,
        BasicOntologyInterface.labels,
        BasicOntologyInterface.curie_to_uri,
        BasicOntologyInterface.uri_to_curie,
        BasicOntologyInterface.ontologies,
        BasicOntologyInterface.obsoletes,
        SearchInterface.basic_search,
        OboGraphInterface.node,
    ]
    """all methods that should be delegated to wrapped_adapter"""

    def __post_init__(self):
        slug = self.resource.slug
        # we delegate to two different implementations
        # 1. an ensmallen_graph
        # 2. a sqlite implementation
        # The selector is one of two forms, indicating whether
        # the grape graph should come directly from KGOBO,
        # or whether it should come from the OAK ontology
        if slug.startswith("kgobo:"):
            # fetch a pre-published ontology simultaneously
            # from kgobo and semsql. Note: there is no guarantee
            # these are in sync, so problems may arise
            slug = slug.replace("kgobo:", "")
            self.uses_biolink = True
            logging.info(f"Fetching {slug} from KGOBO")
            f = get_graph_function_by_name(slug.upper())
            if self.wrapped_adapter is None:
                logging.info(f"Fetching {slug} from SemSQL")
                self.wrapped_adapter = SqlImplementation(OntologyResource(f"obo:{slug.lower()}"))
            self.graph = f(directed=True)
        else:
            # build the grape graph from the OAK ontology
            from oaklib.selector import get_implementation_from_shorthand

            logging.info(f"Wrapping an existing OAK implementation to fetch {slug}")
            inner_oi = get_implementation_from_shorthand(slug)
            self.wrapped_adapter = inner_oi
            self.graph = load_graph_from_adapter(inner_oi)
        self.transposed_graph = self.graph.to_transposed()
        # delegation magic
        methods = dict(inspect.getmembers(self.wrapped_adapter))
        for m in self.delegated_methods:
            mn = m if isinstance(m, str) else m.__name__
            setattr(GrapeImplementation, mn, methods[mn])

    def map_biolink_predicate(self, predicate: PRED_CURIE) -> PRED_CURIE:
        """
        Map from biolink (use in KGX) to RO/OWL.

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
        self.transposed_graph = self.graph.to_transposed()

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
        """Implement OAK interface."""
        g = self.graph
        for n_id in g.get_node_ids():
            yield g.get_node_name_from_node_id(n_id)

    def outgoing_relationships(
        self, curie: CURIE, predicates: List[PRED_CURIE] = None
    ) -> Iterator[Tuple[PRED_CURIE, CURIE]]:
        """Implement OAK interface."""
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
        """Implement OAK interface."""
        return pairs_as_dict(self.outgoing_relationships(*args, **kwargs))

    def incoming_relationships(
        self, curie: CURIE, predicates: List[PRED_CURIE] = None
    ) -> Iterator[Tuple[PRED_CURIE, CURIE]]:
        """Implement OAK interface."""
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

    def incoming_relationship_map(self, *args, **kwargs) -> RELATIONSHIP_MAP:
        """Implement OAK interface."""
        return pairs_as_dict(self.incoming_relationships(*args, **kwargs))

    # -- SemSim methods --

    def pairwise_similarity(
        self,
        subject: CURIE,
        object: CURIE,
        predicates: List[PRED_CURIE] = None,
        subject_ancestors: List[CURIE] = None,
        object_ancestors: List[CURIE] = None,
        counts: dict = None
    ) -> TermPairwiseSimilarity:
        """Implement OAK interface."""
        if predicates:
            raise ValueError("For now can only use hardcoded ensmallen predicates")
        raise NotImplementedError

        if counts is None:
            counts = dict(
                zip(
                    onto_graph.get_node_names(),
                    [1] * len(onto_graph.get_node_names()),
                )
            )
    
        resnik_model = DAGResnik()
        resnik_model.fit(dag, node_counts=counts)
        sim = resnik_model.get_similarities_from_bipartite_graph_node_names(
            source_node_names = [subject],
            target_node_names = [object],
            return_node_names = True
        )

        tp = TermPairwiseSimilarity(
            subject_id=subject,
            object_id=object,
            ancestor_information_content=sim
        )
        return tp


    def predict(self) -> Iterator[Tuple[float, CURIE, Optional[PRED_CURIE], CURIE]]:
        """Implement OAK interface."""
        embedding = FirstOrderLINEEnsmallen().fit_transform(self.graph)
        model = PerceptronEdgePrediction(
            edge_features=None,
            number_of_edges_per_mini_batch=32,
            edge_embeddings="CosineSimilarity",
        )
        model.fit(graph=self.graph, node_features=embedding)
        df = model.predict_proba(
            graph=self.graph, node_features=embedding, return_predictions_dataframe=True
        )
        for _, row in df.iterrows():
            yield row["predictions"], row["sources"], None, row["destinations"]
