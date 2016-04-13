# Tools to process an event graph to extract motifs and other data.

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
    
def find_motifs(event_graph, max_length): 
    global all_motifs # This is bad, real bad
    all_motifs = set()
    # This loop is the easiest to parallelise.
    for node in event_graph.nodes():
        motifs = frozenset({node})
        Vm = set({n for n in event_graph if event_graph.node[n]['id'] < event_graph.node[node]['id']})
        Vp = set({n for n in event_graph if n in event_graph.neighbors(node) and event_graph.node[n]['id'] > event_graph.node[node]['id']})
        subfind(event_graph, max_length, all_motifs, motifs, Vm, Vp)
    return all_motifs