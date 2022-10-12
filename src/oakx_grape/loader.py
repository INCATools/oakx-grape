import csv
import logging
import tempfile

from oaklib import BasicOntologyInterface
from ensmallen import Graph

SUBJECT_COLUMN = "subject"
PREDICATE_COLUMN = "predicate"
OBJECT_COLUMN = "object"
ID_COLUMN = "id"


def load_graph_from_adapter(oi: BasicOntologyInterface, transpose=False, name="Unnamed") -> Graph:
    """
    Creates an ensmallen graph from an OAK ontology interface
    """
    # note: this may be replaced by a kgx writer in oak core
    node_file = tempfile.NamedTemporaryFile("w", newline="", encoding='utf-8', delete=True)
    edge_file = tempfile.NamedTemporaryFile("w", newline="", encoding='utf-8', delete=True)
    logging.info(f"Writing to temp KGX-style node file: {node_file.name}")
    # we avoid using DictWriter as it produces CRLFs which are not parsed by ensmallen
    entities = list(oi.entities(filter_obsoletes=True))
    node_file.write(f"{ID_COLUMN}\n")
    for e in entities:
        node_file.write(e)
        node_file.write("\n")
    logging.info(f"Writing to temp KGX-style edge file:{edge_file.name}")
    edge_file.writelines(f"{SUBJECT_COLUMN}\t{PREDICATE_COLUMN}\t{OBJECT_COLUMN}\n")
    for s, p, o in oi.relationships():
        if s in entities and o in entities and s != o:
            if transpose:
                edge_file.write(f"{o}\t{p}\t{s}\n")
            else:
                edge_file.write(f"{s}\t{p}\t{o}\n")
    node_file.seek(0)
    edge_file.seek(0)
    logging.info(f"Loading files using ensmallen")
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