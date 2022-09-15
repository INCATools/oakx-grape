"""GrapeImplementation test."""
import unittest

from oaklib.selector import get_implementation_from_shorthand
from oakx_grape.loader import load_graph_from_adapter

from tests import TEST_DB


class TestLoader(unittest.TestCase):

    def test_loader(self):
        core_oi = get_implementation_from_shorthand(f"sqlite:{TEST_DB}")
        graph = load_graph_from_adapter(core_oi)
        tg = graph.to_transposed()
        for e in core_oi.entities():
            e_id = tg.get_node_id_from_node_name(e)
            print(f"{e} -> {e_id}")
            for subject_id in tg.get_neighbour_node_ids_from_node_id(e_id):
                subj = tg.get_node_name_from_node_id(subject_id)
                edge_id = tg.get_edge_id_from_node_ids(e_id, subject_id)
                # why doesn't this work?
                pred = tg.get_edge_type_name_from_edge_id(edge_id)
                #pred = None
                print(f"  {subj} {pred} [{edge_id}] {e}")


