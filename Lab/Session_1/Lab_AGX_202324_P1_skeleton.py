import networkx as nx
import pandas as pd
import spotipy
from collections import deque

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def add_node(graph: nx.DiGraph, artist_id: str, sp: spotipy.client.Spotify) -> None:
	artist = sp.artist(artist_id)
	graph.add_node(
		artist_id,
		name=artist['name'],
		genres=artist['genres'][0] if artist['genres'] else "None",
		popularity=artist['popularity'],
		followers=artist['followers']['total']
	)

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


def search_artist(sp: spotipy.client.Spotify, artist_name: str) -> str:
	"""
	Search for an artist in Spotify.

	:param sp: spotipy client object
	:param artist_name: name to search for.
	:return: spotify artist id.
	"""
	# ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

	artist = sp.search(artist_name)
	return artist['tracks']['items'][0]['album']['artists'][0]['id']
	# ----------------- END OF FUNCTION --------------------- #


def crawler(sp: spotipy.client.Spotify, seed: str, max_nodes_to_crawl: int, strategy: str = "BFS",
			out_filename: str = "g.graphml") -> nx.DiGraph:
	"""
	Crawl the Spotify artist graph, following related artists.

	:param sp: spotipy client object
	:param seed: starting artist id.
	:param max_nodes_to_crawl: maximum number of nodes to crawl.
	:param strategy: BFS or DFS.
	:param out_filename: name of the graphml output file.
	:return: networkx directed graph.
	"""
	# ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
	graph = nx.DiGraph()
	visited = set()
	if strategy == "BFS":
		queue = deque([seed])
	else:  # DFS
		queue = [seed]
	
	add_node(graph, seed, sp)
	visited.add(seed)

	# # Continue crawling until the queue/stack is empty or we reach max nodes
	while queue:
		if strategy == "BFS":
			current_artist = queue.popleft()
		else:  # DFS
			current_artist = queue.pop()
		
		try:
			# fetch related artists
			related_artists = sp.artist_related_artists(current_artist)['artists']
		except spotipy.exceptions.SpotifyException as e:
			# in case of network error or similar
			print(f"Error retrieving related artists for {current_artist}: {e}")
			continue

		for artist in related_artists:
			artist_id = artist['id']
			graph.add_edge(current_artist, artist_id)
			if artist_id not in visited:
				# Add new artist to the graph and mark as visited
				add_node(graph, artist_id, sp)
				visited.add(artist_id)
				# Add artist to the queue/stack to further explore
				if len(graph.nodes) < max_nodes_to_crawl:
					queue.append(artist_id)
	
	nx.write_graphml(graph, out_filename)
	return graph
	# ----------------- END OF FUNCTION --------------------- #


def get_track_data(sp: spotipy.client.Spotify, graphs: list, out_filename: str) -> pd.DataFrame:
	"""
	Get track data for each visited artist in the graph.

	:param sp: spotipy client object
	:param graphs: a list of graphs with artists as nodes.
	:param out_filename: name of the csv output file.
	:return: pandas dataframe with track data.
	"""
	# ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
	track_data = []

	# Extract unique artist IDs from the list of graphs
	artist_ids = set()
	for graph in graphs:
		artist_ids.update(graph.nodes)
	artist_ids = list(artist_ids)
	artist_names = artist_names = {artist_id: graph.nodes[artist_id].get('name', 'Unknown')\
								for graph in graphs for artist_id in graph.nodes}
	
	# Fetch top tracks for each artist
	for artist_id in artist_ids:
		try:
			top_tracks = sp.artist_top_tracks(artist_id)['tracks']
		except spotipy.exceptions.SpotifyException as e:
			print(f"Error retrieving top tracks for artist {artist_id}: {e}")
			continue
		
		audios_features = sp.audio_features([track['id'] for track in top_tracks])

		for i, track in enumerate(top_tracks):
			audio_features = audios_features[i]
			track_info = {
				# basic song data
				'track_id': track['id'],
				'duration_ms': track['duration_ms'],
				'track_name': track['name'],
				'popularity': track['popularity'],
				# audio feature data
				'danceability': audio_features['danceability'],
				'energy': audio_features['energy'],
				'loudness': audio_features['loudness'],
				'speechiness': audio_features['speechiness'],
				'acousticness': audio_features['acousticness'],
				'instrumentalness': audio_features['instrumentalness'],
				'liveness': audio_features['liveness'],
				'valence': audio_features['valence'],
				'tempo': audio_features['tempo'],
				# album data
				'album_id': track['album']['id'],
				'album_name': track['album']['name'],
				'release_date': track['album']['release_date'],
				# artist data (only of the artist id)
				'artist_id': artist_id,
				'artist_name': artist_names[artist_id]
			}
			track_data.append(track_info)

	df = pd.DataFrame(track_data)
	df.to_csv(out_filename, index=False)

	return df
	# ----------------- END OF FUNCTION --------------------- #
