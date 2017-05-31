from collections import namedtuple, defaultdict
import networkx as nx

"""
Tools for handling a motif - creation, aggregation, plotting.
"""

class Motif(object):
    """
    A class for temporal motifs.

    Attributes:
        lettermap (dict): A mapping of numbers to letters.
        inv_lettermap (dict): A mapping of letters to numbers.
    """

    lettermap = dict(enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    inv_lettermap = {val: key for key, val in lettermap.items()}

    def __init__(self, iterable, undirected=False, event_format=None):
        """
        Initialises the motif class.

        Args:
            iterable (iterable): Contains a series of events (source, target, time, ...)
            undirected (bool): States whether the motif is undirected (default:False)
            event_format:
            node_colour ():
            edge_color ():
        """

        self._text = None
        self._duration = False
        self._edge_color = False
        self._source_color = False
        self._target_color = False
        self._undirected = undirected

        if isinstance(iterable, str):
            iterable = string_to_iterable(iterable)

        iterable = self._clean_input(iterable, event_format)
        self._extract_motif_information(iterable)

    def _clean_input(self, iterable, event_format=None):
        """
        Cleans input so that it is of the form (source, target, time, duration, type)
        
        Note:
            1. iterable does not need to be ordered.

        Args:
            iterable (iterable): An series of tuples (source, target, time, ...)

        Returns:
            iterable (list): A list of namedtuples.
        """

        iterable = list(iterable)
        if event_format is None:
            event_format = infer_event_type(iterable[0])
        for attribute in event_format[3:]:
            setattr(self, '_' + attribute, True)
        Event = namedtuple('Event', event_format)
        iterable = [Event(*item) for item in iterable]
        iterable.sort(key=lambda x: x.time, reverse=False)  # Sort by timestamp
        return iterable

    def _extract_motif_information(self, iterable):
        """
        Extracts all information from the iterable to create the motif.

        Note:
            Extracts edge durations, edge colors, interevent times and the motif itself.
        
        Args:
            iterable (iterable):

        Returns:
            None
        """

        self.nodemap = {}
        self.eventmarkers = []
        self.motif = []

        ix = 0
        for event in iterable:
            self.eventmarkers.append(event.time)
            if self._duration:  # We might want to just default durations to 0
                self.eventmarkers.append(event.time + event[3])
            if event.source not in self.nodemap.keys():
                self.nodemap[event.source] = ix
                ix += 1
            if event.target not in self.nodemap.keys():
                self.nodemap[event.target] = ix
                ix += 1
            source = self.nodemap[event.source]
            target = self.nodemap[event.target]
            self.motif.append((source, target))

        if self._duration:
            self.durations = [event.duration for event in iterable]
        if self._edge_color:
            self.edgecolors = [event.edge_color for event in iterable]
        if self._source_color and self._target_color:
            self.nodecolors = [(event.source_color, event.target_color) for event in iterable]
            self.nodecolormap = {self.nodemap[event.source]: event.source_color for event in iterable}
            self.nodecolormap.update({self.nodemap[event.target]: event.target_color for event in iterable})

        self.interevent = [j - i for i, j in zip(self.eventmarkers[:-1], self.eventmarkers[1:])][1::2]
        self.original = iterable  # Might need a copy

    @property
    def text(self):
        """str: Returns the standardised text representation of the motif. """
        if self._text is not None: return self._text
        self._text = self._create_motif_text()
        return self._text

    @property
    def type(self):
        """str: Returns the type of motif (directed/colored). """
        attributes = ['_duration', '_edge_color', '_source_color', '_target_color', '_undirected']
        string = '\n'.join(['{} : {}'.format(att[1:], getattr(self, att)) for att in attributes])
        return string

    def _create_motif_text(self):
        """ 
        Constructs the standardised text representation of the motif.

        Note:
            If edges and nodes are plain motifs are of the form (AB-BA).
            If edges are colored motifs returned are of the form (ABx-BCy).
            If edges are undirected then each edge is sorted alphabetically.
            If nodes are colored then motifs are of the form (A1B2-B2A1).

        Args:
            None

        Returns:
            str: the standardised motif text representation.
        """

        if self._edge_color and not self._source_color:
            s = []
            for edge, color in zip(self.motif, self.edgecolors):
                add = self.lettermap[edge[0]] + self.lettermap[edge[1]]
                if self._undirected:
                    add = ''.join(sorted(add))
                add += str(color)
                s.append(add)

        elif self._source_color and not self._edge_color:
            s = []
            for edge, color in zip(self.motif, self.nodecolors):
                add = self.lettermap[edge[0]] + str(color[0]) + self.lettermap[edge[1]] + str(color[1])
                if self._undirected:
                    add = ''.join(sorted(add))
                s.append(add)

        elif self._source_color and self._edge_color:
            s = []
            for edge, ec, nc in zip(self.motif, self.edgecolors, self.nodecolors):
                add = self.lettermap[edge[0]] + str(nc[0]) + self.lettermap[edge[1]] + str(nc[1])
                if self._undirected:
                    add = ''.join(sorted(add))
                add += str(ec)
                s.append(add)

        else:
            s = []
            for edge in self.motif:
                add = self.lettermap[edge[0]] + self.lettermap[edge[1]]
                if self._undirected:
                    add = ''.join(sorted(add))
                s.append(add)

        return '-'.join(s)

    def __repr__(self):
        """ Motif representation. """
        return "<Motif {}>".format(self.text)

    def __str__(self):
        """ Motif string representation. """
        return self.text

    def __hash__(self):
        """ Two motifs are equal if they have the same text representation. """
        return hash(self.__class__) ^ hash(self.text)

    def __eq__(self, other):
        """ Two motifs are equal if they have the same text representation. """
        return isinstance(other, self.__class__) and self.text == other.text

    def __getitem__(self, ix):
        """ Gets the corresponding values from the events which make up the motif. """
        return self.motif[ix]

    def __len__(self):
        """ Returns the length of the motif (number of events). """
        return len(self.motif)

    def __contains__(self, item):
        """ Checks whether a particular sequence is part of a motif using the string implementation. """
        if isinstance(item, str):
            return item.lower() in self.text.lower()
        else:
            raise NotImplementedError("Only strings can be used")

    def to_networkx(self):
        """ 
        Converts the motif to a list of edges and dictionary of edge attributes (edge order, edge type)
        """
        if self._undirected:
            G = nx.Graph(name=self.text)
        else:
            G = nx.DiGraph(name=self.text)

        if self._edge_color:
            edge_attr = [{'t': t+1, 'color': col} for t, col in enumerate(self.edgecolors)]
        else:
            edge_attr = [{'t': t + 1} for t in range(len(self))]
        edges = [(x, y, d) for (x, y), d in zip(self.motif, edge_attr)]
        G.add_edges_from(edges)
        if self._source_color:
            nx.set_node_attributes(G, 'color', self.nodecolormap)
        return G

    def is_valid(self):
        """
        Determines whether a motif is valid (events are consecutive on each node)

        Returns: True if valid, False otherwise.

        """
        ordering = defaultdict(int)
        for event in self.original:
            if ordering[event.source] == 0:
                ordering[event.source] = event.s_order
            elif ordering[event.source] == event.s_order - 1:
                ordering[event.source] += 1
            else:
                return False
            if ordering[event.target] == 0:
                ordering[event.target] = event.t_order
            elif ordering[event.target] == event.t_order - 1:
                ordering[event.target] += 1
            else:
                return False
        return True

def infer_event_type(event):
    """
    Infers the properties of a tuple in the context of events.

    Note:
        Currently not robust.
        Assumes tuple is at least length 3 and of the form (source, target, time, ...).

    Args: 
        event (tuple):

    Returns:
        event_format (list): descriptions for each item in the tuple
    """
    # We might want to relax the string condition on edge and node colors and just convert them ourselves.
    event_format = ['source', 'target', 'time', 'duration']#, 's_order', 't_order', 'dt_index']
    if len(event) == 4:
        return event_format
    elif len(event) == 5:
        if isinstance(event[4], str) or isinstance(event[4], int):
            event_format.append('edge_color')
        return event_format
    elif len(event) == 6:
        if isinstance(event[4], str) and isinstance(event[8], str):  # Missing no dur/color and node colors.
            event_format.extend(['source_color', 'target_color'])
        return event_format
    elif len(event) == 7:
        event_format.extend(['edge_color', 'source_color', 'target_color'])  # Misssing some subcases here.
        return event_format
    else:
        raise InvalidEventTypeError(
            "Events need to be of the form (source, target, time, duration, edge_color, source_color, target_color)")


class InvalidEventTypeError(BaseException):
    """ Raised when an invalid event is passed into a motif. """
    pass


def string_to_iterable(string):
    """
    Converts a standardised motif string into a minimal iterable.

    Note:

    Args:
        string (str): A standardised motif string (e.g. AB-BA)

    Returns:
        iterable (list): A list of events which form the string motif.

    """

    inv_lettermap = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', range(26)))
    events = string.split('-')

    if len(events[0]) == 2:
        iterable = [(inv_lettermap[s[0]], inv_lettermap[s[1]], t) for t, s in enumerate(events)]

    elif len(events[0]) == 3:
        iterable = [(inv_lettermap[s[0]], inv_lettermap[s[1]], t, s[2]) for t, s in enumerate(events)]

    elif len(events[0]) == 4:
        # Node colors
        pass

    # Need to add node colours
    return iterable
