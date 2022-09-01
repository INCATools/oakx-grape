from dataclasses import dataclass
from typing import Iterable

from oaklib import BasicOntologyInterface
import grape
from ensmallen import Graph
from oaklib.types import CURIE


def get_graph_graph_by_name(name: str, module = "kgobo") -> Graph:
    """Dynamically import a Grape class based on its reference.

    :param reference: The reference or path for the class to be imported.
    :return: The imported class
    """
    mod = __import__(f"grape.datasets.{module}", fromlist=[name])
    this_class = getattr(mod, name)
    return this_class()


@dataclass
class GrapeImplementation(BasicOntologyInterface):
    graph: Graph = None

    def __post_init__(self):
        self.graph = get_graph_graph_by_name(self.resource.slug)

    def entities(self, filter_obsoletes=True, owl_type=None) -> Iterable[CURIE]:
        for n in self.graph.get_node_ids():
            yield n

