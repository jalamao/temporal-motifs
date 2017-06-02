import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .auxiliary import *

"""
These functions are copied over from legacy code so are not pretty, but do the job for now.
"""

def draw_tree(tree, dts, min_size=None, *args, **kwargs):
    """
    Draws a heirarchical tree

    Kwargs:
        font_size:
        node_scale:
        figsize:
    """

    def populate_nx_tree(G, tree):
        for element in tree:
            if isinstance(element, Iterable) and not isinstance(element, (int)):
                G.add_node(element, attr_dict={'size':len(list(flatten(element)))})
                for minor_element in element:
                    G.add_edge(element, minor_element)
                populate_nx_tree(G, element)
            else:
                G.add_node(element, attr_dict={'size':1})

    G = nx.DiGraph()           
    populate_nx_tree(G, tree)
    G = nx.convert_node_labels_to_integers(G)

    if min_size is not None:
        remove = [node for node, size in nx.get_node_attributes(G, 'size').items() if size < min_size]
        G.remove_nodes_from(remove)

    # Defaults
    node_scale = kwargs.get('node_scale', 10)
    font_size = kwargs.get('font_size', 6)
    figsize = kwargs.get('figsize', (30,10))
        
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    
    root = [ix for ix, val in nx.in_degree_centrality(G).items() if val == 0.0][0]
    remove = [ix for ix, length in nx.shortest_path_length(G,root).items() if length > len(dts)]
    G.remove_nodes_from(remove)

    dts = sorted(dts)
    dts.append(r'\infty')
    layers = len(dts)
    buffer = 0.2
    spacer = (1 - buffer)/layers
    for ix, dt in enumerate(dts):
        ax.text(0, buffer + ix*spacer, r"$\Delta t = {}$".format(dt), transform=ax.transAxes,
               fontdict={'size':2*font_size})
    
    pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    sizes = nx.get_node_attributes(G, 'size')

    nx.draw_networkx_labels(G, pos, labels=sizes, font_size=font_size, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=False)
    nx.draw_networkx_nodes(G, pos, nodelist=sizes.keys(), node_size=[x*node_scale for x in sizes.values()]);
    
    return None

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

def draw_motif(motif, ax=None, edge_color_map=None, node_color_map=None):
    """
    Draws a single motif.

    Args:
        motif:
        ax:
        edge_color_map:
        node_color_map:

    Returns:

    """

    G = motif.to_networkx()
    layout = nx.circular_layout(G)
    edges = G.edges(data=True)
    nodes = G.nodes(data=True)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if motif._source_color:
        if node_color_map is None:
            node_color = set(list(motif.nodecolormap.values()))
            color = ['grey', 'w', 'darkslategrey', 'lightgrey']
            node_color_map = {col: cm for col, cm in zip(node_color, color)}
        node_color = [node_color_map[node[-1]['color']] for node in nodes]
    else:
        node_color = 'grey'

    if motif._edge_color:
        if edge_color_map is None:
            edge_color = set(motif.edgecolors)
            color = ['g','r','b','orangered','m']
            edge_color_map = {col: cm for col, cm in zip(edge_color, color)}
        edge_color = [edge_color_map[edge[-1]['color']] for edge in edges]
    else:
        edge_color = 'k'

    nx.draw_networkx(G, edges=edges, pos=layout, ax=ax, node_color=node_color, edge_color=edge_color)
    edge_labels = dict([((u, v,), d['t'])
                        for u, v, d in edges])
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=edge_labels,
                                 label_pos=0.25, ax=ax)
    plt.axis('off')
    return None


def draw_motif_distribution(counter, top=3, bottom=3, **kwargs):
    """
    Draws the top and bottom motifs in a counter dictionary.

    Args:
        counter:
        top:
        bottom:
        **kwargs:

    Returns:

    """
    counter={key:val for key,val in counter.items()}

    dist = sorted(counter.items(), key=lambda x: x[1])
    distplot = [x[0] for x in dist[:-(top + 1):-1]] + [x[0] for x in dist[bottom - 1::-1]]
    values = np.asarray([x[1] for x in dist[:-(top + 1):-1]] + [x[1] for x in dist[bottom - 1::-1]], dtype=float)
    total = sum([x[1] for x in dist])

    fig, axes = plt.subplots(1, top + bottom, **kwargs)
    for ax, x, value in zip(axes, distplot, values):
        draw_motif(x, ax)
        ax.set_axis_off()
        ax.text(0.5, -0.25, '{:,} \n ({:.4f}) \n #{}'.format(int(value), value / total, x),
                weight='bold', horizontalalignment='center',
                transform=ax.transAxes)
    return fig