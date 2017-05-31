import sys

from .motif import Motif

"""
# Tools to process an event graph to extract motifs and other data.
"""

def subfind(G, nmax, S_all, S, Vm, Vp):
    """

    Args:
        G:
        nmax:
        S_all:
        S:
        Vm:
        Vp:

    Returns:

    """
    global all_motifs

    for node in Vp:
        Sx = S.union(set({node}))
        all_motifs = all_motifs.union(frozenset({Sx}))
        if len(Sx) == nmax:  # Other functions could go here
            continue
        Vmx = Vm.union(set({x for x in Vp if G.node[x]['id'] < G.node[node]['id']}))
        #Vmx = None
        Vpx = set({x for x in Vm if G.node[x]['id'] > G.node[node]['id']}).union(set({x for x in G.neighbors(node) if x not in Vmx}))
        subfind(G, nmax, S_all, Sx, Vmx, Vpx)
    return None
    
def find_connected_sets(event_graph, max_length, verbose=False):
    """

    Args:
        event_graph:
        max_length:
        verbose:

    Returns:

    """
    global all_motifs # This is bad, real bad
    all_motifs = set()
    # This loop is the easiest to parallelise.
    num_edges = len(event_graph.nodes())
    #adj_list = nx.to_dict_of_lists(event_graph)
    for ix, node in enumerate(event_graph.nodes()):

        if ix % 50 == 0 and verbose:  # This should be moved outside so we dont have to keep checking this.
            sys.stdout.write('\r {}/{}'.format(ix, num_edges))
            sys.stdout.flush()

        motifs = frozenset({node})
        Vm = set({n for n in event_graph if event_graph.node[n]['id'] < event_graph.node[node]['id']})
        #Vm = None
        Vp = set({n for n in event_graph if n in event_graph.neighbors(node) and event_graph.node[n]['id'] > event_graph.node[node]['id']})
        subfind(event_graph, max_length, all_motifs, motifs, Vm, Vp)
    return all_motifs

def find_motifs(event_graph, max_length, verbose=False):
    """
    Returns all motifs in an EventGraph to a maximum length.

    Args:
        event_graph:
        max_length:
        verbose:

    Returns:

    """
    connected_sets = find_connected_sets(event_graph, max_length, verbose)
    valid_subgraphs = [Motif(x) for x in connected_sets if len(x)>1]
    valid_subgraphs = [x for x in valid_subgraphs if x.is_valid()]

    return valid_subgraphs

def is_valid(motif, event_list, event_graph):
    """
    Checks if a motif is valid - events of each node must be consecutive.

    Args:
        motif:
        event_list:
        event_graph:

    Returns:

    """
    ids = sorted([event_graph.node[x]['id'] for x in motif.original])

    intermediate_events = event_list[ids[0]: ids[-1]+1]
    filt = ~intermediate_events.index.isin(ids)
    intermediate_events = intermediate_events[filt]
    for ix, event in intermediate_events.iterrows():
        before = set(event_list[ids[0]:ix][['source', 'target']].values.flatten())
        after = set(event_list[ix+1:ids[-1]+1][['source', 'target']].values.flatten())
        for node in (event.source, event.target):
            if node in before and node in after:
                return False
    return True
