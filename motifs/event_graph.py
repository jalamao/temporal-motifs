# Tools to create an event graph from a temporal network.
import networkx as nx
import pandas as pd

from collections import defaultdict

class EventGraph(nx.DiGraph):
    """
    A class to store temporal events (such as messages, phonecalls) and store them in a causal graph structure.
    """  

    def __init__(self, event_list, *args, **kwargs):
        """ 
        """ 
        super(self.__class__, self).__init__(self, *args, **kwargs)
        self.node = defaultdict(lambda: defaultdict(int)) # This means we can find attributes of nodes which don't exist. 
        # Potential issue when counting the number of nodes we have but address that later.

        if isinstance(event_list, pd.DataFrame):
            self.event_list = event_list
        else:
            try:
                self.event_list = pd.DataFrame(event_list)
            except:
                raise Exception("Cannot convert data to dataframe.") #placeholder.
        return None 

    def build(self):
        """
        """
        for ix, event in enumerate(self.event_list.itertuples(index=False, name='Event')):
            self.add_node(event, defaultdict(int))
            self.node[event]['id'] = ix
            for prev_event in event_filter(ix, event, self.event_list, dt=150):
                if are_connected(prev_event, event) and self.are_neighbours(prev_event, event):
                    self.add_edge(prev_event, event, {'iet': event[2] - (prev_event[2]+prev_event[3])})

                    # This ensures that each node in the event receives or sends one message max. 
                    # This function should be rewritten to be more robust.
                    self.node[prev_event][event[0]] += 2
                    self.node[prev_event][event[1]] += 2
                    self.node[event][prev_event[0]] -= 1
                    self.node[event][prev_event[1]] -= 1 # Some of these will be redundant but I don't think will interfere.

    def are_neighbours(self, e1, e2):
        """
        Checks whether the events are neighbours in the graph
        """
        if (self.node[e1][e2[0]] <= 1) and (self.node[e1][e2[1]] <= 1): 
            return True
        else:
            return False
                    
def event_filter(ix, event, df, dt=99999): # This needs to be generalised so that we can pass in any function, not just dt connectedness.
    """
    Returns an iterable subset of the dataframe.
    """
    time_diff = event[2] - df.time[:ix] # We can make this so that we only look backwards!
    return  df[:ix][time_diff < dt].itertuples(index=False)  

def are_connected(e1, e2):
    """
    Checks whether two events are connected.
    """
    
    if e1 == e2: return False
    #if len(set({e1[0], e2[0]}) & set({e2[0], e2[1]})) > 0:
    if (e1[0] in e2[:2] or e1[1] in e2[:2]):
        return True
    else:
        return False

