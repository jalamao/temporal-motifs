from unittest import TestCase

import pandas as pd
import numpy as np

from motifs import EventGraph

class TestEventBuilding(TestCase):

	def setUp(self):
		self.sample_data = []
		nodes = np.arange(20)
		for i in range(100):
			s1, s2 = np.random.choice(nodes, size=2, replace=False)
			self.sample_data.append((s1,s2,i,0,'r'))
		self.sample_data = pd.DataFrame(self.sample_data, columns=['source', 'target', 'time', 'duration', 'type'])

	def tearDown(self):
		self.sample_data = None

	def test_build(self):
		event_graph = EventGraph(self.sample_data)
		event_graph.build()
		self.assertTrue( max(event_graph.in_degree().values()) <= 2 )
		self.assertTrue( max(event_graph.out_degree().values()) <= 2 )

	def test_are_connected(self):
		pass

