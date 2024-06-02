import networkx as nx
from collections import defaultdict
from networkx.algorithms.community import girvan_newman
from community import community_louvain # pip install python-louvain
from pprint import pprint
import pandas as pd

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def find_mfvs(graph: nx.DiGraph):
	"""
	Find an approximate minimum feedback vertex set for a directed graph.

	:param graph: networkx DiGraph
	:return: set of nodes forming an approximate MFVS
	"""
	g = graph.copy()
	mfvs = set()
	
	# Continue removing nodes until there are no cycles
	while True:
		try:
			cycle = nx.find_cycle(g, orientation='original')
			# Find the node with the highest degree in the cycle
			node_to_remove = max(cycle, key=lambda x: g.degree(x[0]))[0]
			mfvs.add(node_to_remove)
			g.remove_node(node_to_remove)
		except nx.NetworkXNoCycle:
			break
	
	return mfvs

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def num_common_nodes(*args):
	"""
	Return the number of common nodes between a set of graphs.

	:param arg: (an undetermined number of) networkx graphs.
	:return: an integer, number of common nodes.
	"""
	# ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
	if not args:
		return 0
	
	# Extract node sets for each graph
	node_sets = [set(graph.nodes) for graph in args]
	
	# Find intersection
	common_nodes = set.intersection(*node_sets)
	
	return len(common_nodes)
	# ----------------- END OF FUNCTION --------------------- #


def get_degree_distribution(g: nx.Graph) -> dict:
	"""
	Get the degree distribution of the graph.

	:param g: networkx graph.
	:return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
	"""
	# ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
	degree_distribution = defaultdict(int)

	for node, degree in g.degree():
		degree_distribution[degree] += 1

	return dict(degree_distribution)
	# ----------------- END OF FUNCTION --------------------- #


def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
	"""
	Get the k most central nodes in the graph.

	:param g: networkx graph.
	:param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
	:param num_nodes: number of nodes to return.
	:return: list with the top num_nodes nodes.
	"""
	# ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
	if metric == 'degree':
		centrality = nx.degree_centrality(g)
	elif metric == 'betweenness':
		centrality = nx.betweenness_centrality(g)
	elif metric == 'closeness':
		centrality = nx.closeness_centrality(g)
	elif metric == 'eigenvector':
		centrality = nx.eigenvector_centrality(g)
	else:
		raise ValueError(f"Unsupported centrality metric: {metric}")

	# Sort by centrality in descending order
	sorted_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
	
	# Return top num_nodes nodes
	return [node for node, _ in sorted_nodes[:num_nodes]]
	# ----------------- END OF FUNCTION --------------------- #


def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
	"""
	Find cliques in the graph g with size at least min_size_clique.

	:param g: networkx graph.
	:param min_size_clique: minimum size of the cliques to find.
	:return: two-element tuple, list of cliques (each clique is a list of nodes) and
		list of nodes in any of the cliques.
	"""
	# ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
	# Convert graph to undirected
	if g.is_directed():
		g = g.to_undirected()

	# Find all cliques
	cliques = list(nx.find_cliques(g))

	# Filter cliques based on minimum size
	filtered_cliques = [clique for clique in cliques if len(clique) >= min_size_clique]

	# Get nodes in cliques
	nodes_in_cliques = set(node for clique in filtered_cliques for node in clique)

	return filtered_cliques, list(nodes_in_cliques)
	# ----------------- END OF FUNCTION --------------------- #


def detect_communities(g: nx.Graph, method: str, random_seed: int=42) -> tuple:
	"""
	Detect communities in the graph g using the specified method.

	:param g: a networkx graph.
	:param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
	:param random_seed: random seed for louvain method.
	:return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
	"""
	# ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
	# Convert graph to undirected
	if g.is_directed():
		g = g.to_undirected()

	if method == 'girvan-newman':
		comp = girvan_newman(g)
		
		# Extract communities found at first level of the algorithm
		communities = next(comp)
		communities = [list(c) for c in communities]
		
		# Create partition dictionary (node: community_id)
		partition = {}
		for idx, community in enumerate(communities):
			for node in community:
				partition[node] = idx
				
        # Compute modularity (here we use louvain package but doesn't mean we are using louvain method)
		modularity = community_louvain.modularity(partition, g)


	elif method == 'louvain':
		partition = community_louvain.best_partition(g, random_state=random_seed)
		
		# Create communities from partition
		communities = {}
		for node, community_id in partition.items():
			if community_id not in communities:
				communities[community_id] = []
			communities[community_id].append(node)
		communities = list(communities.values())
	
        # Compute modularity
		modularity = community_louvain.modularity(partition, g)
		
	else:
		raise ValueError(f"Unsupported community detection method: {method}")

	return communities, modularity
	# ----------------- END OF FUNCTION --------------------- #


if __name__ == '__main__':
	# ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
	gB = nx.read_graphml('gB.graphml')
	gD = nx.read_graphml('gD.graphml')
	gB2 = nx.read_graphml('gBp.graphml')
	gD2 = nx.read_graphml('gDp.graphml')
	df = pd.read_csv('songs.csv')
	# 1)
	# 	a)
	shared_nodes_BD = num_common_nodes(gB, gD)
	print("Number of common nodes between gB and gD:", shared_nodes_BD,\
	   "(out of", gB.number_of_nodes(), "and", str(gD.number_of_nodes())+")")
	# 	b)
	shared_nodes_BB2 = num_common_nodes(gB, gB2)
	print("Number of common nodes between gB and gB2:", shared_nodes_BB2,\
	   "(out of", gB.number_of_nodes(), "and", str(gB2.number_of_nodes())+")")
	# 2)
	central_25_deg = get_k_most_central(gB2, 'degree', 25)
	central_25_bet = get_k_most_central(gB2, 'betweenness', 25)
	common_central = set(central_25_deg).intersection(central_25_bet)
	# print("25 most central nodes by degree:", central_25_deg)
	# print("25 most central nodes by betweenness:", central_25_bet)
	print(f"Common nodes in both sets: {len(common_central)}")
	# 3)
	min_size_clique_B = 7
	min_size_clique_D = 7
	cliques_B, nodes_in_cliques_B = find_cliques(gB2, min_size_clique_B)
	cliques_D, nodes_in_cliques_D = find_cliques(gD2, min_size_clique_D)
	print(f"Number of cliques in gB2 with at least {min_size_clique_B} nodes:", len(cliques_B))
	print(f"Number of cliques in gD2 with at least {min_size_clique_D} nodes:", len(cliques_D))
	print(f"Number of nodes part of the cliques in gB2:", len(nodes_in_cliques_B))
	print(f"Number of nodes part of the cliques in gD2:", len(nodes_in_cliques_D))
	# 4)
	max_size_clique_B = max(cliques_B, key=len)
	# get their graph nodes and print characteristics
	max_size_clique_B = [gB.nodes[i] for i in max_size_clique_B]
	print("Nodes in the largest clique in gB2:")
	pprint(max_size_clique_B)
	# 5)
	communities, modularity = detect_communities(gD, 'louvain', random_seed=42)
	print(f"Modularity of the partition found by Louvain in gD: {modularity}")
	# 6)
		# a)
	mfvs_B = find_mfvs(gB)
	mfvs_D = find_mfvs(gD)
	print(f"The minimum cost for graph gB is {len(mfvs_B)} x 100 = {len(mfvs_B)*100}")
	print(f"The minimum cost for graph gD is {len(mfvs_D)} x 100 = {len(mfvs_D)*100}")
	# 	b)
	betweenness_centrality = nx.betweenness_centrality(gB)
	sorted_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
	top_nodes = [node for node, _ in sorted_nodes[:4]]
	print("Top 4 nodes with highest betweenness centrality in gB:")
	pprint([gB.nodes[node]["name"] for node in top_nodes])
	betweenness_centrality = nx.betweenness_centrality(gD)
	sorted_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
	top_nodes = [node for node, _ in sorted_nodes[:4]]
	print("Top 4 nodes with highest betweenness centrality in gD:")
	pprint([gD.nodes[node]["name"] for node in top_nodes])
	# 7)
	meghan = df[df['artist_name'] == 'Meghan Trainor']["artist_id"].values[0]
	kali = df[df['artist_name'] == 'Kali Uchis']["artist_id"].values[0]
	path = nx.shortest_path(gB, source=meghan, target=kali)
	print("Shortest path between Meghan Trainor and Kali Uchis in gB:")
	pprint([gB.nodes[node]["name"] for node in path])

	# ------------------- END OF MAIN ------------------------ #
