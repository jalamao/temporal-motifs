# Tools for handling a motif - creation, aggregation, plotting.
from collections import namedtuple


lettermap = dict(enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
inv_lettermap = {val: key for key, val in lettermap.items()}

class Motif(object):

    lettermap = dict(enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    inv_lettermap = {val: key for key, val in lettermap.items()}
    
    def __init__(self, iterable, undirected=False,
                                 duration=False,
                                 edge_color=False,
                                 source_color=False,
                                 target_color=False,
                                 node=None):
        """
        Class: Motif
        
        """

        self._text = None
        self.node = node
        self._duration = duration
        self._edge_color = edge_color
        self._source_color = source_color
        self._target_color = target_color
        self._undirected = undirected

        if isinstance(iterable, basestring):
            iterable = string_to_iterable(iterable)

        iterable = self._clean_input(iterable)
        self._process_input(iterable)
        


    def _clean_input(self, iterable):
        """
        Cleans input so that it is of the form (source, target, time, duration, type)
        """
        iterable = list(iterable)
        event_format = infer_event_type(iterable[0])
        for attribute in event_format[3:]:
            setattr(self, '_'+attribute, True)
        Event = namedtuple('Event', event_format)
        iterable = [Event(*item) for item in iterable]
        iterable.sort(key = lambda x: x.time, reverse=False) # Sort by timestamp
        return iterable
    
    def _process_input(self, iterable):
        """
        Processes an iterable into a motif.
        Extracts edge durations, edge colors, interevent times and the motif itself.
        """
        
        self.nodemap = {}
        self.eventmarkers = []
        self.motif = []
        
        if self._duration:
            self.durations = [event.duration for event in iterable]
        if self._edge_color:
            self.edgecolors = [event.edge_color for event in iterable]
        if self._source_color and self._target_color:
            self.nodecolormap = {} # TBC

        ix = 0
        for event in iterable:
            self.eventmarkers.append(event.time)
            if self._duration: # We might want to just default durations to 0
                self.eventmarkers.append(event.time+event[3])
            if event.source not in self.nodemap.keys():
                self.nodemap[event.source] = ix
                ix += 1
            if event.target not in self.nodemap.keys():
                self.nodemap[event.target] = ix
                ix += 1
            source = self.nodemap[event.source]
            target = self.nodemap[event.target]
            self.motif.append((source,target))

        self.interevent = [j-i for i, j in zip(self.eventmarkers[:-1], self.eventmarkers[1:])][1::2]
        self.original = iterable # Might need a copy
        
    @property
    def text(self):
        """ Returns the standardised text representation of the motif. """
        if self._text is not None: return self._text
        self._text = self._create_motif_text()
        return self._text

    @property
    def type(self):
        """ Returns the type of motif. """

        attributes = ['_duration', '_edge_color', '_source_color', '_target_color', '_undirected']
        string = '\n'.join(['{} : {}'.format(att[1:], getattr(self, att)) for att in attributes])
        return string

    def _create_motif_text(self):
        """ 
        Returns the standardised text representation of the motif.
        If edges are colored motifs returned are of the form [ABxBCy].
        If edges are undirected then each edge is sorted alphabetically.
        """
        
        if self._edge_color:
            s = []
            for edge, color in zip(self.motif, self.edgecolors):
                add = self.lettermap[edge[0]] + self.lettermap[edge[1]]
                if self._undirected:
                    add = ''.join(sorted(add))
                add += color
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
        """ Motif string representation. """
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
        """ Returns the length of the motif. """
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
        return self.motif, dict(zip(self.motif, range(1,len(self)+1)))

def infer_event_type(event):
    
    event_format = ['source', 'target', 'time']
    if len(event)==3:
        return event_format
    elif len(event)==4:
        if isinstance(event[3], basestring):
            event_format.append('edge_color')
        else:
            event_format.append('duration')
        return event_format
    elif len(event)==5:
        if isinstance(event[3], basestring) and isinstance(event[4], basestring): # Missing no dur/color and node colors.
            event_format.extend(['source_color', 'target_color'])
        elif isinstance(event[3], basestring):
            event_format.extend(['edge_color', 'duration'])
        else:
            event_format.extend(['duration', 'edge_color'])
        return event_format
    elif len(event)==7:
        event_format.extend(['duration', 'edge_color', 'source_color', 'target_color']) # Misssing some subcases here.
        return event_format
    else:
        raise InvalidEventTypeError("Events need to be of the form (source, target, time, duration, edge_color, source_color, target_color)")

class InvalidEventTypeError(BaseException):
    """ """
    pass


def string_to_iterable(string):
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
