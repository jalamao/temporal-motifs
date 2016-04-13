# Tools to create an event graph from a temporal network.
import networkx as nx

class EventGraph(nx.DiGraph):
    """
    A class to store temporal events (such as messages, phonecalls) and store them in a causal graph structure.
    """  

    def __init__(self):
    	return None 


def are_connected(G, e1, e2, dt=999999):
    """
    Checks whether two events are dt connected in the event_graph, G.
    e1 should be the earlier event.
    """
    
    if e1 == e2: return False
    if (e1[0] in e2[:2] or e1[1] in e2[:2]) and (e2[2] - (e1[2]+e1[3]) < dt):
        if (G.node[e1].get(e2[0], 0) <= 1) and (G.node[e1].get(e2[1], 0) <= 1): # Really want a defaultdict for the node attributes
            return True
        else:
            return False

def build_event_graph(event_list):
    event_graph = nx.DiGraph()
    for ix, event in enumerate(event_list):
        event_graph.add_node(event, {'id':ix, event[0]:0, event[1]:0})
        for prev_event in event_graph.nodes():     
        # It would be better to iterates through a sliced dataframe "through edges which are less than dt old"
            if are_connected(event_graph, prev_event, event):
                event_graph.add_edge(prev_event, event, {'time': event[2] - (prev_event[2]+prev_event[3])})

                try:
                    event_graph.node[prev_event][event[0]] += 2
                except KeyError:
                    pass
                try:
                    event_graph.node[prev_event][event[1]] += 2
                except KeyError:
                    pass
                try:
                    event_graph.node[event][prev_event[0]] -= 1
                except KeyError:
                    pass
                try:
                    event_graph.node[event][prev_event[1]] -= 1 # Some of these will be redundant but I don't think will interfere.
                except KeyError:
                    pass

                #print("Addding edge", prev_event, event, {'time': event[2] - (prev_event[2]+prev_event[3])})
    return event_graph
