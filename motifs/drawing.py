import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

"""
These functions are copied over from legacy code so are not pretty, but do the job for now.
"""

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