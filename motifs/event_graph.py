import sys
from bisect import bisect_left
from collections import defaultdict, namedtuple
from itertools import count

import networkx as nx
import numpy as np
import pandas as pd

from .motif import Motif

"""
Tools to create an event graph from a temporal network.
"""

class EventGraph(nx.DiGraph):
    """
    A class to store temporal events (such as messages, phonecalls) and store them in a causal graph structure.
    """

    @classmethod
    def from_pickle(cls, filename):
        import pickle
        with open(filename, 'rb') as file:
            event_graph = pickle.load(file)
        #Event = namedtuple('Event', event_graph._event_fields)
        #event_graph = nx.relabel_nodes(event_graph, {node: Event(*node) for node in event_graph.nodes()})
        return event_graph

    @classmethod
    def from_eventlist(cls, event_list, dt, *args, **kwargs):
        event_graph = cls(*args, **kwargs)
        #event_graph.node = defaultdict(lambda: defaultdict(int)) # This means we can find attributes of nodes which don't exist. 
        # Potential issue when counting the number of nodes we have but address that later.
        event_graph.dt = dt

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

        #event_graph.event_list.source = event_graph.event_list.source.astype(int)
        #event_graph.event_list.target = event_graph.event_list.target.astype(int)
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

        #lbs = np.zeros(shape=(len(event_graph.event_list), 1), dtype=int)
        #for ix, entry in enumerate(event_graph.event_list.time - event_graph.dt):
        #    lbs[ix] = bisect_left(event_graph.event_list.time, entry)
        #event_graph.event_list['lower_index'] = lbs

        # Partition the data into bins corresponding to dt, then we need only to slice the data rather than compare times.
        #groups = pd.cut(event_graph.event_list.time, bins=np.arange(0, event_graph.event_list.time.max()+dt, dt))
        #event_graph.partition = [0] + list(event_graph.event_list[groups != groups.shift(1)].index)
        return event_graph

    @classmethod
    def from_filter(cls, teg, dt, inc_nodes=True, *args, **kwargs):
        event_graph = cls(*args, **kwargs)
        if inc_nodes:
            event_graph.add_nodes_from(teg.nodes(data=True))
        edges = [edge for edge in teg.edges(data=True) if edge[2]['iet'] <= dt]
        event_graph.add_edges_from(edges)
        #for edge, iet in [(edge,iet) for edge, iet in nx.get_edge_attributes(teg, 'iet').items() if iet < dt]:
        #    event_graph.add_edge(*edge, attr_dict={'iet': iet})
        try:  # nx.connected_component_subgraphs doesn't copy all the information. Assume if we're using it then we don't need event list.
            event_graph.event_list = teg.event_list.ix[event_graph.nodes()]
        except:
            pass
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

        Args:
            verbose (bool) -

        Returns:

        """

        node_pointer = defaultdict(lambda: None) # Points to the index of the last event for that node
        #position = ['s_next', 't_next']
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

        # OLD
        # num_events = len(self.event_list)
        # for ix, event in enumerate(self.event_list.itertuples(index=False, name='Event')):

        #     if ix % 50 == 0 and verbose: # This should be moved outside so we don't have to keep checking this.
        #         sys.stdout.write('\r {}/{}'.format(ix, num_events))
        #         sys.stdout.flush()

        #     self.add_node(event, defaultdict(bool))
        #     self.node[event]['id'] = ix
        #     self.node[event]['count'] = 0
        #     for prev_event in self.event_filter(ix, event, self.event_list, dt=self.dt):

        #         if self.node[prev_event]['count'] >= 2: continue  # If the node already has 2 edges, skip it .

        #         shared = list(set(prev_event[:2]) & set(event[:2]))
        #         if (are_connected(prev_event, event) and self.are_neighbours(prev_event, event)) or len(shared) == 2:

        #             if len(shared) == 2 and self.node[prev_event][shared[0]] != self.node[prev_event][shared[1]]:
        #                 # print ("Repeated nodes ABBA or ABAB")
        #                 self.node[prev_event][shared[0]] = True
        #                 self.node[prev_event][shared[1]] = True
        #                 self.node[prev_event]['count'] += 1
        #                 self.add_edge(prev_event, event, {'iet': event[2] - (prev_event[2] + prev_event[3])})

        #             self.add_edge(prev_event, event, {'iet': event[2] - (prev_event[2]+prev_event[3])})  #we probably should just add ints as nodes
        #             for node in shared:
        #                 self.node[prev_event][node] = True
        #                 self.node[prev_event]['count'] += 1

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

    def event_filter(self, ix, event, df, dt=99999): # This needs to be generalised so that we can pass in any function, not just dt connectedness.
        """
        Returns an iterable subset of the dataframe.

        Args:
            ix:
            event:
            df:
            dt:

        Returns:

        """
        # try:
        #     lower_bound = self.partition[max(0, bisect_left(self.partition, ix)-2)]
        # except IndexError as e:
        #     lower_bound = 0
        lower_bound = event[-1] # lower_bound in event_list
        time_diff = event[2] - df.time[lower_bound:ix]  # We can make this so that we only look backwards!
        return df[lower_bound:ix][time_diff < dt].itertuples(index=False, name='Event')

    @property
    def maximal_subgraphs(self):
        """
        Lists the weakly connected components of the event graph in descending order of size.

        Returns:
            list
        """
        return sorted(nx.weakly_connected_components(self), key=len, reverse=True)

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

    def to_edgelist(self, time_indexed=False, durations=False):
        """
        Converts the event graph into an edgelist.
        Node enumeration starts at 0.
        Starting time of all components is 0 unless times are given.

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
        return self.get_motif_distribution(valid_only=True)

    def get_motif_distribution(self, valid_only=True):
        """
        Returns the distribution of motifs in the TEG.
        Valid_only picks only motifs which are valid (by definition of Kovanen et al.)
        """
        if valid_only:
            edges = {edge: val for edge, val in nx.get_edge_attributes(self, 'type').items() if not val.endswith('i')}.keys()
            motifs = [self[u][v]['type'] for (u,v) in edges]
        else:
            motifs = list(nx.get_edge_attributes(self, 'type').values())
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

    def save(self, filename):
        """
        Saves the event_graph as a pickle.

        Args:
            filename:
        """
        #new_event_graph = nx.relabel_nodes(self, {x: x for x in self.nodes()}) # Removes the lambda from the nodes
        self._event_fields = self.event_list.columns[:-4]
        import pickle
        with open(filename, 'wb') as file:
            pickle.dump(self, file, -1)


def are_connected(e1, e2):
    """
    Checks whether two events are connected.

    Args:
        e1:
        e2:

    Returns:

    """
    
    if e1 == e2: return False
    if (e1[0] in e2[:2] or e1[1] in e2[:2]):
        return True
    else:
        return False


def plot_full_barcode_efficiently(teg, dt, top, ax):
    """ Prints a barcode. """
    import matplotlib
    
    filtered = teg.filter_edges(dt)
    segs = []
    tmin, tmax = 1e99, 0
    for ix,component in enumerate(sorted(nx.weakly_connected_components(filtered), key=len, reverse=True)[:top]):
        component = [teg.event_list.ix[i] for i in component]
        for event in component:
            segs.append(((event.time, ix),(event.time, ix+1)))
            tmax = max(tmax, event.time)
            tmin = min(tmin, event.time)
            
    ln_coll = matplotlib.collections.LineCollection(segs, linewidths=1, colors='k')
    bc = ax.add_collection(ln_coll)
    ax.set_ylim((0, top+1))
    ax.set_xlim((tmin,tmax))   
    return bc

def tweets_to_edgelist(df):
    """ 
    Converts a set of tweets into a set of events between users. 
    Takes only the first mention in each tweet.
    """
    event_list = []
    ix = 0
    df = df.sort_index()
    for _, row in df.iterrows():
        
        source = row['user_name']
        tweet_id = row['id_str']
        
        if len(row['entities']['user_mentions']) > 0:
            target = row['entities']['user_mentions'][0]['screen_name']
        else:
            continue

        time = int(row['created_at'].value/int(1e9))
        
        if row['in_reply_to_status_id_str'] is not None:
            style = 'reply'
        elif row['retweeted_status_id_str'] is not None:
            style = 'retweet'
            source, target = target, source
        else:
            style = 'message'
            
        event_list.append((source, target, time, style, tweet_id))

    event_list = pd.DataFrame(event_list, columns=['source','target','time','edge_color', 'tweet_id'])    
    return event_list