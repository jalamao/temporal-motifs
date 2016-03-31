# Tools for handling a motif - creation, aggregation, plotting.

class Motif(object):


    lettermap = dict(zip(range(26), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    
    def __init__(self, iterable, colored_edges=False, 
                                 colored_nodes=False,
                                 undirected=False,
                                 node=None):
        """
        Class: Motif
        
        """
        iterable = self._clean_input(iterable)
        self._process_input(iterable)
        
        self._text = None
        self.colored_edges = colored_edges
        self.colored_nodes = colored_nodes
        self.undirected = undirected

    def _clean_input(self, iterable):
        """
        Cleans input so that it is of the form (source, target, time, duration, type)
        """
        iterable = list(iterable)
        iterable.sort(key=lambda x:x[2], reverse=False) # sort by timestamp
        
        return iterable
    
    def _process_input(self, iterable):
        """
        Processes an iterable into a motif.
        Extracts edge durations, edge colors, interevent times and the motif itself.
        """
        
        self.nodemap = {}
        self.durations = []
        self.eventmarkers = []
        self.edgecolors = []
        self.motif = []
        ix = 0
        for event in iterable:
            self.durations.append(event[3])
            self.eventmarkers.append(event[2])
            self.eventmarkers.append(event[2]+event[3])
            self.edgecolors.append(event[4])
            if event[0] not in self.nodemap.keys():
                self.nodemap[event[0]] = ix
                ix += 1
            if event[1] not in self.nodemap.keys():
                self.nodemap[event[1]] = ix
                ix += 1
            source = self.nodemap[event[0]]
            target = self.nodemap[event[1]]
            self.motif.append((source,target))

        self.interevent = [j-i for i, j in zip(self.eventmarkers[:-1], self.eventmarkers[1:])][1::2]
        self.original = iterable.copy()
        
    @property
    def text(self):
        """ Returns the standardised text representation of the motif. """
        if self._text is not None: return self._text
        self._text = self._create_motif_text()
        return self._text
    
    def _create_motif_text(self):
        """ 
        Returns the standardised text representation of the motif.
        If edges are colored motifs returned are of the form [ABxBCy].
        If edges are undirected then each edge is sorted alphabetically.
        """
        
        if self.colored_edges:
            s = ''
            for edge, color in zip(self.motif, self.edgecolors):
                add = self.lettermap[edge[0]] + self.lettermap[edge[1]]
                if self.undirected:
                    s += ''.join(sorted(add))
                else:
                    s += add
                s += color
        else: 
            s = ''
            for edge in self.motif:
                add = self.lettermap[edge[0]] + self.lettermap[edge[1]]
                if self.undirected:
                    s += ''.join(sorted(add))
                else:
                    s += add
        return s
        
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