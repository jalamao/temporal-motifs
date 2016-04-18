# Dynamical decomposition of a network.

from collections import defaultdict
import networkx as nx

# 1. Split motifs into fast and slow (ideally we want any arbitrary partition of the IETs).
# Tail chosen as the last 50% of value/probability.

#We'll just look at ALL IETs for now
def find_tail_value(motifs):
	iets = []
	for motif in motifs:
		iets.append(motif.interevent)
	iets = np.array(iets)
	median = np.median(iets)
	return median

# 2. Collect all motifs of particular types (slow ABBA e.t.c).

def partition_motifs(motifs, split):

	collect = defaultdict(list)
	
	def spliter(motif):
		if motif.interevent < split:
			return "fast"
		else:
			return "slow"

	for motif in motifs:
		collect[motif.text + " ({})".format(splitter(motif))].append(motif)

	return collect

# 3. For each partition, add the edges of each event to the layer.

def build_networks(partition):

	layers = {}
	for key, motifs in partition.items():
		G = nx.DiGraph(name=key)
		for motif in motifs:
			for event in motif.original:
				G.add_edge(event.source, event.target)
		layers[key] = G

# 4. Build the multilayer network from each motif layer.
