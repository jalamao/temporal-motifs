# Tools to process an event graph to extract motifs and other data.
from .motif import Motif

def subfind(G, nmax, S_all, S, Vm, Vp):
    global all_motifs
    all_motifs = all_motifs.union(frozenset({S}))
    if len(S) == nmax: # Other functions could go here
        return None
    for node in Vp:
        Sx = S.union(set({node}))
        Vmx = Vm.union(set({x for x in Vp if G.node[x]['id'] < G.node[node]['id']}))
        Vpx = set({x for x in Vm if G.node[x]['id'] > G.node[node]['id']}).union(set({x for x in G.neighbors(node) if x not in Vmx}))
        subfind(G, nmax, S_all, Sx, Vmx, Vpx)
    return None
    
def find_connected_sets(event_graph, max_length): 
    global all_motifs # This is bad, real bad
    all_motifs = set()
    # This loop is the easiest to parallelise.
    for node in event_graph.nodes():
        motifs = frozenset({node})
        Vm = set({n for n in event_graph if event_graph.node[n]['id'] < event_graph.node[node]['id']})
        Vp = set({n for n in event_graph if n in event_graph.neighbors(node) and event_graph.node[n]['id'] > event_graph.node[node]['id']})
        subfind(event_graph, max_length, all_motifs, motifs, Vm, Vp)
    return all_motifs

def find_motifs(event_list, event_graph, max_length):
    """ Returns all motifs in an EventGraph to a maximum length."""
    connected_sets = find_connected_sets(event_graph, max_length)
    valid_subgraphs = [Motif(x) for x in connected_sets if len(x)>1]

    if max_length > 2: # All length 2 motifs are automatically valid.
        valid_subgraphs = [x for x in valid_subgraphs if is_valid(x, event_list, event_graph)]

    return valid_subgraphs

def is_valid(motif, event_list, event_graph):
    """ Checks if a motif is valid - events of each node must be consecutive. """
    ids = [event_graph.node[x]['id'] for x in motif.original]

    intermediate_events = event_list[ids[0]: ids[-1]+1]
    filt = ~intermediate_events.index.isin(ids)
    intermediate_events = intermediate_events[filt]
    for ix, event in intermediate_events.iterrows():
        before = set(event_list[ids[0]:ix][['source','target']].values.flatten())
        after = set(event_list[ix+1:ids[-1]+1][['source','target']].values.flatten())
        for node in (event.source, event.target):
            if node in before and node in after:
                return False
    return True