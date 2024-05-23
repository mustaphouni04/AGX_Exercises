import networkx as nx
import pandas as pd
import os
import random

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    # create an empty undirected graph
    undirected_graph = nx.Graph()

    # iterate over all edges in the directed graph
    for u, v in g.edges():
        # check if the reverse edge also exists in the directed graph
        if g.has_edge(v, u):
            # add the edge to the undirected graph
            undirected_graph.add_edge(u, v)

    # save the undirected graph in graphml format
    nx.write_graphml(undirected_graph, out_filename)

    print("Current working directory:", os.getcwd())
    if os.path.exists('bidirectional_graph.graphml'):
        print("File written successfully!")
    else:
        print("Failed to write file.")

    return undirected_graph

# Create a directed graph
g = nx.DiGraph()
g.add_edge('A', 'B')
g.add_edge('B', 'A')
g.add_edge('A', 'C')
g.add_edge('C', 'D')
g.add_edge('D', 'C')

# Call the function
undirected_graph = retrieve_bidirectional_edges(g, 'bidirectional_graph.graphml')

print("Nodes in the undirected graph:", undirected_graph.nodes())
print("Edges in the undirected graph:", undirected_graph.edges())

def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # create a copy of the original graph
    pruned_graph = g.copy()

    # remove nodes with degree less than min_degree
    nodes_to_remove = [node for node, degree in pruned_graph.degree() if degree < min_degree]
    pruned_graph.remove_nodes_from(nodes_to_remove)

    # remove zero-degree/isolated nodes
    pruned_graph.remove_nodes_from([node for node, degree in pruned_graph.degree() if degree == 0])

    # save the pruned graph to a file
    nx.write_graphml(pruned_graph, out_filename)

    return pruned_graph

g = nx.gnm_random_graph(100, 1000)
pruned_graph = prune_low_degree_nodes(g, 25, 'pruned_graph.graphml')

print("Nodes in the pruned graph:", pruned_graph.nodes())
print("Edges in the pruned graph:", pruned_graph.edges())


def prune_low_weight_edges(g: nx.Graph, min_weight=None, min_percentile=None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    if (min_weight is None and min_percentile is None) or (min_weight is not None and min_percentile is not None):
        raise ValueError("Exactly one of min_weight or min_percentile must be specified")

    pruned_graph = g.copy()

    if min_weight is not None:
        threshold = min_weight
    else:
        weights = [edata['weight'] for u, v, edata in g.edges(data=True)]
        threshold = np.percentile(weights, min_percentile)

    edges_to_remove = [(u, v) for u, v, edata in pruned_graph.edges(data=True) if edata['weight'] < threshold]
    pruned_graph.remove_edges_from(edges_to_remove)

    pruned_graph.remove_nodes_from([node for node, degree in pruned_graph.degree() if degree == 0])

    if out_filename is not None:
        nx.write_graphml(pruned_graph, out_filename)

    return pruned_graph

# create a random graph with 10 nodes and 15 edges
g = nx.gnm_random_graph(10, 15)

# add random weights to the edges
for u, v in g.edges():
    g[u][v]['weight'] = random.uniform(0, 1)

# print the original graph details
print("Original graph:")
print("Nodes:", g.number_of_nodes())
print("Edges:", g.number_of_edges())

# Prune the graph by removing edges with weight < 0.5
pruned_graph = prune_low_weight_edges(g, min_weight=0.5, out_filename="pruned_graph.graphml")

# Print the pruned graph details
print("\nPruned graph:")
print("Nodes:", pruned_graph.number_of_nodes())
print("Edges:", pruned_graph.number_of_edges())

def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    pass
    # ----------------- END OF FUNCTION --------------------- #


def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> \
        nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    pass
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    pass
    # ------------------- END OF MAIN ------------------------ #
