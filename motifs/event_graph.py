import sys
from bisect import bisect_left
from collections import defaultdict, namedtuple, Counter
from itertools import count

import networkx as nx
import numpy as np
import pandas as pd

from .motif import Motif
from .auxiliary import *

"""
Tools to create an event graph from a temporal network.
"""

class EventGraph(nx.DiGraph):
	"""
	A class to store temporal events (such as messages, phonecalls) and store them in a causal graph structure.
	"""

	@classmethod
	def from_pickle(cls, filename):
		""" Load an event graph from a previously saved graph. """

		import pickle
		with open(filename, 'rb') as file:
			event_graph = pickle.load(file)
		return event_graph

	@classmethod
	def from_eventlist(cls, event_list, *args, **kwargs):
		""" Load an event graph from a list of events. Need to call EventGraph.build() to build the graph. """

		event_graph = cls(*args, **kwargs)

		if isinstance(event_list, pd.DataFrame):
			event_graph.event_list = event_list
		else:
			try:
				event_graph.event_list = pd.DataFrame(event_list, columns=['source', 'target', 'time', 'duration'][:len(event_list[0])])
			except:
				raise Exception("Cannot convert data to Pandas Dataframe.") #placeholder.

		# Ensure Dataframe is of the form 'source', 'target', 'time, 'duration' + other columns
		columns = event_graph.event_list.columns.tolist()
		if 'duration' not in columns:
			event_graph.event_list.loc[:,'duration'] = 0

		essential = ['source', 'target', 'time', 'duration']
		others = [x for x in columns if x not in essential]
		event_graph.event_list = event_graph.event_list[essential + others]

		if 's_order' not in columns or 't_order' not in columns:
			event_graph.event_list.loc[:,'s_order'] = 0
			event_graph.event_list.loc[:,'t_order'] = 0
		if 's_next' not in columns or 't_next' not in columns:
			event_graph.event_list.loc[:,'s_next'] = None # We'd like to keep this as an in column ideally for optimisation
			event_graph.event_list.loc[:,'t_next'] = None

		event_graph.event_list.index = event_graph.event_list.index.astype(int)

		# Calculate node event orderings for validity checks
		ordering = defaultdict(int)
		extra = np.zeros(shape=(len(event_graph.event_list), 2), dtype=int)
		for ix, row in event_graph.event_list.iterrows():
			ordering[row.source] += 1
			ordering[row.target] += 1
			extra[ix, :] = ordering[row.source], ordering[row.target]
		event_graph.event_list.loc[:,'s_order'] = extra[:, 0]
		event_graph.event_list.loc[:,'t_order'] = extra[:, 1]

		return event_graph

	@classmethod
	def from_filter(cls, teg, dt, inc_nodes=True, *args, **kwargs):
		""" Create an event graph by filtering the edges from another event graph. """

		event_graph = cls(*args, **kwargs)
		edges = [edge for edge in teg.edges(data=True) if edge[2]['iet'] <= dt]
		event_graph.add_edges_from(edges)

		try: 
			event_graph.event_list = teg.event_list.ix[event_graph.nodes()]
		except:
			pass

		if inc_nodes:
			event_graph.add_nodes_from(teg.nodes(data=True))

		return event_graph

	def __init__(self, *args, **kwargs):
		"""

		Args:
			event_list:
			dt
			*args:
			**kwargs:
		"""
		super(self.__class__, self).__init__(self, *args, **kwargs)
		

	def build(self, verbose=False):
		"""
		Build the EventGraph from an event list (adds inter-event times but not motifs to edges).

		Args:
			verbose (bool) - Print out progress

		Returns:

		"""

		node_pointer = defaultdict(lambda: None) # Points to the index of the last event for that node
		position = [list(self.event_list.columns).index('s_next'), list(self.event_list.columns).index('t_next')]
		num_events = len(self.event_list)

		edges = []
		
		for ix, event in self.event_list.iterrows():

			if ix % 50 == 0 and verbose: # This should be moved outside so we don't have to keep checking this.
				sys.stdout.write('\r {}/{}'.format(ix, num_events))
				sys.stdout.flush()

			# Ex and Ey are the previous events for the source and target. 
			ex, ey = node_pointer.get(event.source), node_pointer.get(event.target)  

			if ex is None and ey is None:
				pass

			if ex is not None and ey is not None:

				self.event_list.iat[ex[0], position[ex[1]]] = ix 
				self.event_list.iat[ey[0], position[ey[1]]] = ix

				previous_event = self.event_list.iloc[ex[0]]
				if (ex[0] == ey[0]):
					edges.append((ex[0], ix, event.time - (previous_event.time + previous_event.duration)))
				else:
					edges.append((ex[0], ix, event.time - (previous_event.time + previous_event.duration)))
					edges.append((ey[0], ix, event.time - (previous_event.time + previous_event.duration)))

			if ex is not None:
				self.event_list.iat[ex[0], position[ex[1]]] = ix 

				previous_event = self.event_list.iloc[ex[0]]
				edges.append((ex[0], ix, event.time - (previous_event.time + previous_event.duration)))
			if ey is not None:
				self.event_list.iat[ey[0], position[ey[1]]] = ix

				previous_event = self.event_list.iloc[ey[0]]
				edges.append((ey[0], ix, event.time - (previous_event.time + previous_event.duration)))

			node_pointer[event.source] = (ix, 0)
			node_pointer[event.target] = (ix, 1)

		self.add_nodes_from(self.event_list.index)
		self.add_weighted_edges_from(edges, weight='iet')

	def are_neighbours(self, e1, e2):
		"""
		Checks whether the events are neighbours in the graph

		Args:
			e1:
			e2:

		Returns:

		"""
		if not self.node[e1][e2[0]] and not self.node[e1][e2[1]]:
			return True
		else:
			return False

	@property
	def maximal_subgraphs(self):
		"""
		Lists the weakly connected components of the event graph in descending order of size.

		Returns:
			list
		"""
		return sorted(nx.weakly_connected_component_subgraphs(self), key=len, reverse=True)

	def filter_edges(self, dt):
		"""
		Filters the edges to remove any with an inter-event time greater than dt.

		Args:
			dt:

		Returns:
			nx.Digraph
		"""

		return EventGraph.from_filter(self, dt)

	def add_edge_types(self, verbose=False, columns=None, check_validity=True):
		"""
		Adds the edge attribute 'type' to edges in the network.
		This corresponds to the 2-event motif pattern between the two nodes.

		Args:
			verbose (bool): If true, prints iteration.

		Returns: None
		"""

		if columns is None:
			columns = self.event_list.columns[:-4] #This does not work, ['source', 'target', 'time']

		if 's_order' not in columns:
			columns.append('s_order')
		if 't_order' not in columns:
			columns.append('t_order')
		
		if verbose:
			num_edges = len(self.edges())
			for ix, edge in enumerate(self.edges()):

				sys.stdout.write('\r {}/{}'.format(ix, num_edges))
				sys.stdout.flush()
				events = [tuple(self.event_list.loc[i, columns]) for i in edge]
				m = Motif(events, event_format=columns)

				if check_validity and not m.is_valid():
					self[edge[0]][edge[1]]['type'] = m.text + 'i'
				else:
					self[edge[0]][edge[1]]['type'] = m.text

		else:
			for edge in self.edges():
				events = [tuple(self.event_list.loc[i, columns]) for i in edge]
				m = Motif(events, event_format=columns)
				
				if check_validity and not m.is_valid():
					self[edge[0]][edge[1]]['type'] = m.text + 'i'
				else:
					self[edge[0]][edge[1]]['type'] = m.text

		return None

	def to_event_list(self, time_indexed=False, durations=False):
		"""
		Converts the event graph into an edgelist.
		Node enumeration starts at 0.
		Starting time of all components is 0 unless times are given.
		[Currently a useless function as we are already building event graphs from event lists]

		Args:
			time_indexed (bool):
			durations (bool):

		Returns:
			event_list (list):
		"""
		if durations or not time_indexed:
			raise NotImplementedError

		events = {}
		indexer = count(0)
		# 1. For each node, if it is a root, add event
		# 2. For each node, learn nodes from in-edges.
		# 3. Add the IET from previous event.
		# 4. If not time_indexed, sort out backpropogation.

		if time_indexed:
			for node in sorted(self.nodes(), key=lambda x: x.time):
				in_nodes = self.predecessors(node)
				if len(in_nodes) == 0:
					events[node] = (next(indexer), next(indexer), node.time)  # New event
					continue

				u, v = None, None
				for pred in in_nodes:
					edge_type = self[pred][node]['type']
					# t = pred.time + self[pred][node]['iet']
					t = node.time
					if edge_type == 'AB-BC':
						u = events[pred][1]  # target
					elif edge_type == 'AB-CB':
						v = events[pred][1]
					elif edge_type == 'AB-AC':
						u = events[pred][0]  # source
					elif edge_type == 'AB-CA':
						v = events[pred][0]
					elif edge_type == 'AB-AB':
						u, v = events[pred][0], events[pred][1]
					elif edge_type == 'AB-BA':
						u, v = events[pred][1], events[pred][0]

				if u is None:
					u = next(indexer)
				if v is None:
					v = next(indexer)

				events[node] = (u, v, t)

		events = sorted(list(events.values()), key=lambda x: x[2])
		return events

	@property
	def motif_distribution(self):
		return self.get_motif_distribution(valid_only=True, aggregate=True)

	def get_motif_distribution(self, valid_only=True, aggregate=False):
		"""
		Returns the distribution of motifs in the TEG.
		Valid_only picks only motifs which are valid (by definition of Kovanen et al.)
		"""

		if valid_only:
			edges = {edge: val for edge, val in nx.get_edge_attributes(self, 'type').items() if not val.endswith('i')}.keys()
			motifs = [self[u][v]['type'] for (u,v) in edges]
		else:
			motifs = list(nx.get_edge_attributes(self, 'type').values())

		if aggregate: motifs = Counter(motifs)

		return motifs

	@property
	def iet_distribution(self):
		return self.get_iet_distribution(valid_only=True)

	def get_iet_distribution(self, valid_only=True, by_motif=False):
		"""
		Returns the distribution of IETs in the TEG.
		Valid_only picks only IETs which are part of valid motifs (by definition of Kovanen et al.)
		"""

		if valid_only and not by_motif:
			edges = {edge: val for edge, val in nx.get_edge_attributes(self, 'type').items() if not val.endswith('i')}.keys()
			iets = [self[u][v]['iet'] for (u,v) in edges]
		elif by_motif:
			edges = {edge: val for edge, val in nx.get_edge_attributes(self, 'type').items() if not val.endswith('i')}
			iets = defaultdict(list)
			[iets[motif].append(self[u][v]['iet']) for (u,v), motif in edges.items()]
		else:
			iets = list(nx.get_edge_attributes(self, 'iet').values())
		return iets

	def build_tree(self, dts):
		"""
		Creates a list-of-list tree representation showing the recursive temporal decomposition of the network.

		Args:
			teg:
			dts:
		"""  
		  
		dts = sorted(dts, reverse=True)
		dts.insert(0, 1e99)
		
		def component_tree(teg, dts):
			components = nx.weakly_connected_component_subgraphs(teg.filter_edges(dt=dts[0]))
			branch = []
			for component in components:

				if len(component) == 1:
					branch.append(component.nodes()[0])
				elif len(dts) == 1:
					branch.append(tuple(component.nodes()))
				else:
					branch.append(tuple(component_tree(component, dts[1:])))
			return branch
		
		tree = tuple(component_tree(self, dts))
		self.tree = tree
		return tree

	def save(self, filename):
		"""
		Saves the event_graph as a pickle.

		Args:
			filename:
		"""

		self._event_fields = self.event_list.columns[:-4]
		import pickle
		with open(filename, 'wb') as file:
			pickle.dump(self, file, -1)
		return None