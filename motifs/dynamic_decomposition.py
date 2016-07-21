# Dynamical decomposition of a network.

from collections import defaultdict

import networkx as nx
import pandas as pd
import numpy as np

"""
"""

# 1. Split motifs into fast and slow (ideally we want any arbitrary partition of the IETs).
# Tail chosen as the last 50% of value/probability.

# We'll just look at ALL IETs for now
def find_tail_value(motifs):
    """

    Args:
        motifs:

    Returns:

    """
    iets = []
    for motif in motifs:
        iets.append(motif.interevent)
    iets = np.array(iets)
    median = np.median(iets)
    return median


# 2. Collect all motifs of particular types (slow ABBA e.t.c).

def partition_motifs(motifs, split):
    """

    Args:
        motifs:
        split:

    Returns:

    """
    collect = defaultdict(list)

    def spliter(motif):
        if motif.interevent < split:
            return "fast"
        else:
            return "slow"

    for motif in motifs:
        collect[motif.text + " ({})".format(splitter(motif))].append(motif)

    return collect


def partition_network(motifs):
    """ Partitions motifs into events.

    Note:
        Extra to "partition_motifs" and will be combined later.

    Args:
        motifs

    Returns:
        Temporal networks partitioned by motif.
    """
    collect = defaultdict(list)

    for motif in motifs:
        collect[motif.text].extend(motif.original)
    for key, item in collect.items():
        df = pd.DataFrame.from_records(item)
        df = df.sort_values(by=2).reset_index(drop=True)
        collect[key] = df
    return collect


# 3. For each partition, add the edges of each event to the layer.

def build_networks(partition):
    """

    Args:
        partition:

    Returns:

    """
    layers = {}
    for key, motifs in partition.items():
        G = nx.DiGraph(name=key)
        for motif in motifs:
            for event in motif.original:
                G.add_edge(event.source, event.target)
        layers[key] = G

# 4. Build the multilayer network from each motif layer.
