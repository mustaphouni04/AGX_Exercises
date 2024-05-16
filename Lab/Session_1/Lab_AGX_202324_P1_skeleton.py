import networkx as nx
import pandas as pd
import spotipy
from collections import deque

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


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
    
    graph.add_node(seed)
    visited.add(seed)

    while queue and len(graph.nodes) < max_nodes_to_crawl:
        if strategy == "BFS":
            current_artist = queue.popleft()
        else:  # DFS
            current_artist = queue.pop()
        
        try:
            related_artists = sp.artist_related_artists(current_artist)['artists']
        except spotipy.exceptions.SpotifyException as e:
            print(f"Error retrieving related artists for {current_artist}: {e}")
            continue

        for artist in related_artists:
            artist_id = artist['id']
            if artist_id not in visited:
                graph.add_node(artist_id)
                graph.add_edge(current_artist, artist_id)
                visited.add(artist_id)
                if len(graph.nodes) >= max_nodes_to_crawl:
                    break
                if strategy == "BFS":
                    queue.append(artist_id)
                else:  # DFS
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
    # Initialize an empty list to store track data
    track_data = []

    # Extract unique artist IDs from the list of graphs
    artist_ids = set()
    for graph in graphs:
        artist_ids.update(graph.nodes)
    
    # Fetch top tracks for each artist
    for artist_id in artist_ids:
        try:
            top_tracks = sp.artist_top_tracks(artist_id)['tracks']
        except spotipy.exceptions.SpotifyException as e:
            print(f"Error retrieving top tracks for artist {artist_id}: {e}")
            continue

        for track in top_tracks:
            track_info = {
                'artist_id': artist_id,
                'artist_name': track['artists'][0]['name'],
                'track_id': track['id'],
                'track_name': track['name'],
                'album_name': track['album']['name'],
                'release_date': track['album']['release_date'],
                'popularity': track['popularity']
            }
            track_data.append(track_info)

    # Create a DataFrame from the track data
    df = pd.DataFrame(track_data)

    # Save the DataFrame to a CSV file
    df.to_csv(out_filename, index=False)

    return df
    # ----------------- END OF FUNCTION --------------------- #
