import csv
import tempfile

from oaklib import BasicOntologyInterface
from ensmallen import Graph


def load_graph_from_adapter(oi: BasicOntologyInterface) -> Graph:
    node_file = tempfile.NamedTemporaryFile("w", newline="", encoding='utf-8', delete=False)
    edge_file = tempfile.NamedTemporaryFile("w", newline="", encoding='utf-8', delete=False)
    w = csv.DictWriter(node_file, fieldnames=["id"], delimiter="\t", dialect='unix')
    w.writeheader()
    entities = list(oi.entities(filter_obsoletes=True))
    for e in entities:
        w.writerow({"id": e})
        #print(e)
    w = csv.DictWriter(edge_file, fieldnames=["subject", "predicate", "object"], delimiter="\t", dialect='unix')
    w.writeheader()
    for s, p, o in oi.relationships():
        if s in entities and o in entities and s != o:
            row = {"subject": s, "predicate": p, "object": o}
            w.writerow(row)
            #print(f"ROW={row}")
    node_file.seek(0)
    edge_file.seek(0)
    print(node_file.name)
    g = Graph.from_csv(
        edge_path=edge_file.name,
        edge_list_separator="\t",
        edge_list_header=True,
        sources_column="subject",
        destinations_column="object",
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