
import numpy as np
import pandas as pd

import unittest
from unittest import TestCase

from motifs import EventGraph, find_motifs

class BaseTestCase(TestCase):
        
    def setUp(self):
        """ Generates a sample (random) temporal network to work with."""
        self.sample_data = []
        nodes = np.arange(20)
        for i in range(100):
            s1, s2 = np.random.choice(nodes, size=2, replace=False)
            self.sample_data.append((s1, s2, i, 0, 'r'))
        self.sample_data = pd.DataFrame(self.sample_data, columns=['source', 'target', 'time', 'duration', 'type'])
        self.event_graph = EventGraph.from_eventlist(self.sample_data)
        self.event_graph.build()

    def tearDown(self):
        self.sample_data = None
        self.event_graph = None
        
class TestMotifFinder(BaseTestCase):
    
    def test_finder_function(self):
        """ Ensure function runs. """
        motifs = find_motifs(self.event_graph, 100, 3, columns=['source','target','time'])
        
    def test_coloured_finder_function(self):
        """ Ensure function runs with event types. """
        motifs = find_motifs(self.event_graph, 100, 3, columns=['source','target','time', 'type'])

if __name__ == "__main__":
    unittest.main()