"""GrapeImplementation test."""
import unittest

from ensmallen import Graph

from tests import TEST_EDGES_TSV, TEST_NODES_TSV


class TestGrapeImplementation(unittest.TestCase):
    """Test GrapeImplementation."""

    def test_load(self):
        """Test loading from ensmallen (temp)."""
        Graph.from_csv(
            # Edges related parameters
            # The path to the edges list tsv
            edge_path=str(TEST_EDGES_TSV),
            # Set the tab as the separator between values
            edge_list_separator="\t",
            # The first rows should be used as the columns names
            edge_list_header=True,
            sources_column="subject",
            destinations_column="object",
            # Both source and destinations columns use node names
            edge_list_numeric_node_ids=False,
            edge_list_edge_types_column="predicate",
            # Nodes related parameters
            # The path to the nodes list tsv
            node_path=str(TEST_NODES_TSV),
            # Set the tab as the separator between values
            node_list_separator="\t",
            # KGX
            node_list_header=True,
            nodes_column="id",
            default_node_type="biolink:NamedThing",
            # Graph related parameters
            # The graph is undirected
            directed=False,
            # The name of the graph is HomoSapiens
            name="HomoSapiens",
            # Display a progress bar, (this might be in the terminal and not in the notebook)
            verbose=False,
        )
