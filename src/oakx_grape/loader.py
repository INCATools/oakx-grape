"""Loads ensmallen graphs."""
import logging
import tempfile

from ensmallen import Graph
from oaklib import BasicOntologyInterface

SUBJECT_COLUMN = "subject"
PREDICATE_COLUMN = "predicate"
OBJECT_COLUMN = "object"
ID_COLUMN = "id"


def load_graph_from_adapter(oi: BasicOntologyInterface, transpose=False, name="Unnamed") -> Graph:
    """
    Create an ensmallen graph from an OAK ontology interface.

    This currently works by serializing the OAK graph as minimal KGX,
    and using Grape/ensmallen to load.
    """
    # note: this may be replaced by a kgx writer in oak core
    node_file = tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=True)
    edge_file = tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=True)
    logging.info(f"Writing to temp KGX-style node file: {node_file.name}")
    # we avoid using DictWriter as it produces CRLFs which are not parsed by ensmallen
    entities = set(list(oi.entities(filter_obsoletes=False)))

    logging.info(f"Writing to temp KGX-style edge file:{edge_file.name}")
    edge_file.writelines(f"{SUBJECT_COLUMN}\t{PREDICATE_COLUMN}\t{OBJECT_COLUMN}\n")
    for s, p, o in oi.relationships():
        if (
            s != o
            and s is not None
            and o is not None
            and not s.startswith("_")
            and not o.startswith("_")
            and o in entities
        ):
            if transpose:
                edge_file.write(f"{o}\t{p}\t{s}\n")
            else:
                edge_file.write(f"{s}\t{p}\t{o}\n")
    node_file.write(f"{ID_COLUMN}\n")
    for e in entities:
        node_file.write(e)
        node_file.write("\n")
    node_file.seek(0)
    edge_file.seek(0)
    logging.info(f"Loading files using ensmallen, path={edge_file.name}")
    g = Graph.from_csv(
        edge_path=edge_file.name,
        edge_list_separator="\t",
        edge_list_header=True,
        sources_column=SUBJECT_COLUMN,
        destinations_column=OBJECT_COLUMN,
        edge_list_edge_types_column=PREDICATE_COLUMN,
        edge_list_numeric_node_ids=False,
        node_path=node_file.name,
        node_list_separator="\t",
        node_list_header=True,
        nodes_column="id",
        directed=True,
        name=name,
        verbose=False,
    )
    return g
