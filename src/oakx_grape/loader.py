import csv
import logging
import tempfile

from oaklib import BasicOntologyInterface
from ensmallen import Graph


def load_graph_from_adapter(oi: BasicOntologyInterface, transpose=False) -> Graph:
    """
    Creates an ensmallen graph from an OAK ontology interface
    """
    # note: this may be replaced by a kgx writer in oak core
    node_file = tempfile.NamedTemporaryFile("w", newline="", encoding='utf-8', delete=True)
    edge_file = tempfile.NamedTemporaryFile("w", newline="", encoding='utf-8', delete=True)
    # we avoid using DictWriter as it produces CRLFs which are not parsed by ensmallen
    entities = list(oi.entities(filter_obsoletes=True))
    node_file.write("id\n")
    for e in entities:
        node_file.write(e)
        node_file.write("\n")
    edge_file.writelines("subject\tpredicate\tobject\n")
    for s, p, o in oi.relationships():
        if s in entities and o in entities and s != o:
            if transpose:
                edge_file.write(f"{o}\t{p}\t{s}\n")
            else:
                edge_file.write(f"{s}\t{p}\t{o}\n")
            #row = {"subject": s, "predicate": p, "object": o}
            #w.writerow(row)
            #print(f"ROW={s} {p} {o}")
    node_file.seek(0)
    edge_file.seek(0)
    logging.debug(f"Saved nodes to {node_file.name}")
    logging.debug(f"Saved edges to {edge_file.name}")
    g = Graph.from_csv(
        edge_path=edge_file.name,
        edge_list_separator="\t",
        edge_list_header=True,
        sources_column="subject",
        destinations_column="object",
        edge_types_column="predicate",
        edge_list_numeric_node_ids=False,
        node_path=node_file.name,
        node_list_separator="\t",
        node_list_header=True,
        nodes_column="id",
        directed=True,
        name="MyTest",
        verbose=False,
    )
    return g