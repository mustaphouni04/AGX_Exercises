import networkx as nx
from collections import defaultdict
from networkx.algorithms.community import girvan_newman
from community import community_louvain # pip install python-louvain

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


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


def detect_communities(g: nx.Graph, method: str) -> tuple:
	"""
	Detect communities in the graph g using the specified method.

	:param g: a networkx graph.
	:param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
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
		partition = community_louvain.best_partition(g)
		
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
	pass
	# ------------------- END OF MAIN ------------------------ #
