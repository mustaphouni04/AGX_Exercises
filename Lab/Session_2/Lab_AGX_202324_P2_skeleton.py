import networkx as nx
import pandas as pd
import os
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

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
			# get the weight of the edge in the directed graph
			weight = g[u][v].get('weight', 1)  # default weight is 1 if not found

			# add the edge to the undirected graph with its weight
			undirected_graph.add_edge(u, v, weight=weight)

	# save the undirected graph in graphml format
	nx.write_graphml(undirected_graph, out_filename)

	return undirected_graph


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

def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Compute the mean audio features for tracks of the same artist.

	:param tracks_df: tracks dataframe (with audio features per each track).
	:return: artist dataframe (with mean audio features per each artist).
	"""
	# group the tracks by artist and compute the mean audio features
	artist_df = tracks_df.groupby('artist_id').agg({
		'danceability': 'mean',
		'energy': 'mean',
		'loudness': 'mean',
		'speechiness': 'mean',
		'acousticness': 'mean',
		'instrumentalness': 'mean',
		'liveness': 'mean',
		'valence': 'mean',
		'tempo': 'mean'
	}).reset_index()

	# add the artist name to the resulting dataframe
	artist_df['artist_name'] = artist_df['artist_id'].map(tracks_df.set_index('artist_id')['artist_name'].to_dict())

	return artist_df

def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> nx.Graph:
	"""
	Create a similarity graph from a dataframe with mean audio features per artist.

	:param artist_audio_features_df: dataframe with mean audio features per artist.
	:param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
	:param out_filename: name of the file that will be saved.
	:return: a networkx graph with the similarity between artists as edge weights.
	"""
	# create a new graph
	G = nx.Graph()

	# add nodes for each artist
	for index, row in artist_audio_features_df.iterrows():
		G.add_node(row['artist_id'], name=row['artist_name'])

	# compute the similarity matrix
	if similarity == "cosine":
		similarity_matrix = cosine_similarity(artist_audio_features_df.drop(['artist_id', 'artist_name'], axis=1))
	elif similarity == "euclidean":
		similarity_matrix = 1 / (1 + euclidean_distances(artist_audio_features_df.drop(['artist_id', 'artist_name'], axis=1)))
	else:
		raise ValueError("Invalid similarity metric")

	# add edges with similarity as weight
	for i in range(len(artist_audio_features_df)):
		for j in range(i+1, len(artist_audio_features_df)):
			G.add_edge(artist_audio_features_df.iloc[i]['artist_id'], artist_audio_features_df.iloc[j]['artist_id'], weight=similarity_matrix[i, j])

	# save the graph to a file
	if out_filename is not None:
		nx.write_graphml(G, out_filename)

	return G


if __name__ == "__main__":
	# 6
	gB = nx.read_graphml('../Session_1/gB.graphml')
	gD = nx.read_graphml('../Session_1/gD.graphml')

	# a)
	undirected_gB = retrieve_bidirectional_edges(gB, "gBp.graphml")
	undirected_gD = retrieve_bidirectional_edges(gD, "gDp.graphml")
	print(undirected_gB)
	print(undirected_gD)

	# b)
	df = pd.read_csv("../Session_1/songs.csv")
	df = compute_mean_audio_features(df)
	gW = create_similarity_graph(df, "euclidean", "gw.graphml")
	print(gW)
	
	pruned_graph = prune_low_weight_edges(gW, min_percentile=90)
	num_nodes = pruned_graph.number_of_edges()
	print(f"Number of nodes after pruning at 90th percentile: {num_nodes}")
	
	# 1.3
	# 1)
	# calculate the number of weakly and strongly connected components in gB
	num_weakly_connected_gB = nx.number_weakly_connected_components(gB)
	num_strongly_connected_gB = nx.number_strongly_connected_components(gB)
	# calculate the number of weakly and strongly connected components in gD
	num_weakly_connected_gD = nx.number_weakly_connected_components(gD)
	num_strongly_connected_gD = nx.number_strongly_connected_components(gD)
	# print the results
	print(f"gB - Number of weakly connected components: {num_weakly_connected_gB}") # 1
	print(f"gB - Number of strongly connected components: {num_strongly_connected_gB}") # 372
	print(f"gD - Number of weakly connected components: {num_weakly_connected_gD}") # 1
	print(f"gD - Number of strongly connected components: {num_strongly_connected_gD}") # 393

	# 2)
	# calculate the number of weakly and strongly connected components in gB
	num_connected_gB_undirected = nx.number_connected_components(undirected_gB)
	# calculate the number of weakly and strongly connected components in gD
	num_connected_gD_undirected = nx.number_connected_components(undirected_gD)
	# print the results
	print(f"gB - Number of weakly connected components: {num_connected_gB_undirected}") # 1
	print(f"gD - Number of weakly connected components: {num_connected_gD_undirected}") # 1

	# 3)
	# get the edge weights as a dictionary
	edge_weights = nx.get_edge_attributes(gW, 'weight')
	# find the two most similar artists
	most_similar_artists = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)[:1]
	print("Most similar artists:")
	for edge, weight in most_similar_artists:
		artist_name1 = gW.nodes[edge[0]]['name']
		artist_name2 = gW.nodes[edge[1]]['name']
		print(f"Artist {artist_name1} and Artist {artist_name2} with similarity {weight}")
	# find the two least similar artists
	least_similar_artists = sorted(edge_weights.items(), key=lambda x: x[1])[:1]
	print("Least similar artists:")
	for edge, weight in least_similar_artists:
		artist_name1 = gW.nodes[edge[0]]['name']
		artist_name2 = gW.nodes[edge[1]]['name']
		print(f"Artist {artist_name1} and Artist {artist_name2} with similarity {weight}")
