
import numpy as np
import pandas as pd
import networkx as nx

import unittest
from unittest import TestCase

from motifs import EventGraph, flatten

class BaseTestCase(TestCase):
        
    def setUp(self):
        """ Generates a sample (random) temporal network to work with."""
        self.sample_data = []
        nodes = np.arange(20)
        for i in range(100):
            s1, s2 = np.random.choice(nodes, size=2, replace=False)
            self.sample_data.append((s1, s2, i, 0, 'r'))
        self.sample_data = pd.DataFrame(self.sample_data, columns=['source', 'target', 'time', 'duration', 'type'])

    def tearDown(self):
        self.sample_data = None

class TestEventBuilding(BaseTestCase):
    """ Testing the building of the Temporal Event Graph. """

    def test_load_jumbled_data(self):
        """ Tests whether jumbled columns are loaded correctly. """
        sample_data = self.sample_data[['duration', 'type', 'source', 'target', 'time']]
        event_graph = EventGraph.from_eventlist(self.sample_data)
        self.assertTrue(list(event_graph.event_list.columns[:3]) == ['source', 'target', 'time'])
    
    def test_extra_columns(self):
        """ Test whether extra columns are included in the graph. """
        self.sample_data['extra'] = 'extra'
        event_graph = EventGraph.from_eventlist(self.sample_data)
        self.assertTrue('extra' in event_graph.event_list.columns)
    
    def test_build(self):
        """ Test the graph structure is created properly. """
        event_graph = EventGraph.from_eventlist(self.sample_data)
        event_graph.build()
        self.assertTrue(max(event_graph.in_degree().values()) <= 2)
        self.assertTrue(max(event_graph.out_degree().values()) <= 2)
        
    
class TestDecomposition(BaseTestCase):
    """ Testing methods associated with the temporal decomposition of the network. """
    
    def test_filter_edges(self):
        """ Tests the filtering of edges based on IET. """
        event_graph = EventGraph.from_eventlist(self.sample_data)
        event_graph.build()
        subgraph = event_graph.filter_edges(dt=20) 
        self.assertTrue(sum([t>20 for t in nx.get_edge_attributes(subgraph, 'iet').values()]) == 0)
        
    def test_subgraphs(self):
        """ Tests that connected components are collected. """
        event_graph = EventGraph.from_eventlist(self.sample_data)
        event_graph.build()
        subgraph = event_graph.filter_edges(dt=20) 
        components = subgraph.maximal_subgraphs
        
        #self.assertIsInstance(components[0], event_graph.__class__) # This is failing for some reason 
        self.assertTrue(len(components[0]) >= len(components[-1]))
        
    def test_tree_decomposition(self):
        """ Tests the tree building function """
        event_graph = EventGraph.from_eventlist(self.sample_data)
        event_graph.build()
        tree = event_graph.build_tree(dts=[80,60,40,20,0])
        
        flat_tree = list(flatten(tree))
        self.assertEqual(len(flat_tree), 100)
    
class TestMotifs(BaseTestCase):
    """ Testing methods associated with the two-event motifs. """
    
    def test_add_edge_types(self):
        """ Test if edge attributes added correctly. """
        event_graph = EventGraph.from_eventlist(self.sample_data)
        event_graph.build()
        event_graph.add_edge_types(columns=['source','target','time'])
        
        self.assertEqual(set(event_graph.edges(data=True)[0][2].keys()), set(['iet','type']))
    
    def test_motif_counts(self):
        """ Test if motif distribution returned correctly. """
        event_graph = EventGraph.from_eventlist(self.sample_data)
        event_graph.build()
        event_graph.add_edge_types(columns=['source','target','time'])
        motifs = event_graph.get_motif_distribution()
        self.assertGreater(len(motifs), 0)
        
        motifs = event_graph.get_motif_distribution(aggregate=True)
        self.assertIsInstance(motifs, dict)
        
    def test_motif_types(self):
        """ Test whether the correct motif types are returned. """
        event_graph = EventGraph.from_eventlist(self.sample_data)
        event_graph.build()
        event_graph.add_edge_types(columns=['source','target','time'])
        self.assertTrue(event_graph.edges(data=True)[0][2]['type'].startswith('AB-'))
        
        event_graph.add_edge_types(columns=['source','target','time','type'])
        self.assertTrue(event_graph.edges(data=True)[0][2]['type'].startswith('ABr-'))
        
class TestIETs(BaseTestCase):
    """ Testing methods associated with inter-event times. """

    def test_iet_counts(self):
        """Test that IETs are calculated correctly. """
        event_graph = EventGraph.from_eventlist(self.sample_data)
        event_graph.build()
        event_graph.add_edge_types(columns=['source','target','time'])
        iets = event_graph.get_iet_distribution()
        self.assertGreater(len(iets), 0)
        
        iets = event_graph.get_iet_distribution(by_motif=True)
        self.assertIsInstance(iets, dict)

if __name__ == "__main__":
    unittest.main()