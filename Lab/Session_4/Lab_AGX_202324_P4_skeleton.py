import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import networkx as nx
import numpy as np
import seaborn as sns
from sklearn.metrics import pairwise

# Add the parent directories to the sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Session_3.Lab_AGX_202324_P3_skeleton import get_degree_distribution
from Session_2.Lab_AGX_202324_P2_skeleton import compute_mean_audio_features, prune_low_weight_edges

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def get_most_similar_artist(gw: nx.Graph, artist_id: str):
    # get the edge weights as a dictionary
    edge_weights = nx.get_edge_attributes(gw, 'weight')

    # find the most similar artist
    taylor_swift_edges = [(u, v, d) for (u, v), d in edge_weights.items() if u == artist_id or v == artist_id]
    most_similar_artist_edge = max(taylor_swift_edges, key=lambda x: x[2])

    if most_similar_artist_edge[0] == artist_id:
        most_similar_artist = most_similar_artist_edge[1]
    else:
        most_similar_artist = most_similar_artist_edge[0]

    return most_similar_artist

def get_least_similar_artist(gw: nx.Graph, artist_id: str):
    # get the edge weights as a dictionary
    edge_weights = nx.get_edge_attributes(gw, 'weight')
    
    # find the least similar artist to Taylor Swift
    taylor_swift_edges = [(u, v, d) for (u, v), d in edge_weights.items() if u == artist_id or v == artist_id]
    least_similar_artist_edge = min(taylor_swift_edges, key=lambda x: x[2])
    
    if least_similar_artist_edge[0] == artist_id:
        least_similar_artist = least_similar_artist_edge[1]
    else:
        least_similar_artist = least_similar_artist_edge[0]

    return least_similar_artist

def prune_and_plot(g, min_percentile_range):
    node_counts = []
    for min_percentile in min_percentile_range:
        pruned_graph = prune_low_weight_edges(g, min_percentile=min_percentile)
        connected_components = list(nx.connected_components(pruned_graph))
        largest_component = max(connected_components, key=len)
        node_counts.append(len(largest_component))

    plt.plot(min_percentile_range, node_counts)
    plt.xlabel('Min Percentile')
    plt.ylabel('Number of Nodes in Largest Connected Component')
    plt.title('Number of Nodes as they`re being pruned')
    plt.show()

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def plot_degree_distribution(degree_dict: dict, normalized: bool = False, loglog: bool = False) -> None:
    """
    Plot degree distribution from dictionary of degree counts.

    :param degree_dict: dictionary of degree counts (keys are degrees, values are occurrences).
    :param normalized: boolean indicating whether to plot absolute counts or probabilities.
    :param loglog: boolean indicating whether to plot in log-log scale.
    """
    degrees = list(degree_dict.keys())
    counts = list(degree_dict.values())
    if normalized:
        counts = [count / sum(counts) for count in counts]
    plt.bar(degrees, counts, color = "orange")
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
    plt.show()



def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str) -> None:
    """
    Plot a single figure with a plot of mean audio features of two different artists.

    :param artists_audio_feat: dataframe with mean audio features of artists.
    :param artist1_id: string with id of artist 1.
    :param artist2_id: string with id of artist 2.
    :return: None
    """
    # Filter the dataframe to get the rows for the two artists
    artist1_row = artists_audio_feat[artists_audio_feat['artist_id'] == artist1_id]
    artist2_row = artists_audio_feat[artists_audio_feat['artist_id'] == artist2_id]

    # Select only the numeric columns
    numeric_cols = artists_audio_feat.select_dtypes(include=[int, float]).columns

    # Extract the audio features from the two rows
    artist1_features = artist1_row[numeric_cols].values[0]
    artist2_features = artist2_row[numeric_cols].values[0]

    # Get the artist names
    artist1_name = artist1_row['artist_name'].values[0]
    artist2_name = artist2_row['artist_name'].values[0]

    # Create a figure with a single plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the audio features as bar charts
    bar_width = 0.35
    opacity = 0.8

    index = range(len(artist1_features))

    ax.bar([x - bar_width/2 for x in index], artist1_features, bar_width, alpha=opacity, label=artist1_name, log = True)
    ax.bar([x + bar_width/2 for x in index], artist2_features, bar_width, alpha=opacity, label=artist2_name, log = True)

    # Set the x-axis tick labels
    ax.set_xticks(range(len(artist1_features)))
    ax.set_xticklabels(numeric_cols, rotation=90)

    # Set the title and labels
    ax.set_title("Audio features comparison")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Log Value")

    # Add a legend
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.

    :param artist_audio_features_df: dataframe with mean audio features of artists.
    :param similarity: string with similarity measure to use.
    :param out_filename: name of the file to save the plot. If None, the plot is not saved.
    """
    # Select only the audio features columns (excluding artist_id and artist_name)
    audio_features_df = artist_audio_features_df.iloc[:, 1:-1]

    # Select only the first 15 artists
    audio_features_df = audio_features_df.head(15)
    artist_names = artist_audio_features_df['artist_name'].head(15)

    # Compute the similarity matrix
    if similarity == 'manhattan':
        similarity_matrix = pairwise.pairwise_distances(audio_features_df, metric='manhattan')
        similarity_matrix = 1 - similarity_matrix
    if similarity == 'euclidean':
        similarity_matrix = pairwise.pairwise_distances(audio_features_df, metric='euclidean')
        similarity_matrix = 1 - similarity_matrix

    # Round the values in the similarity matrix to one decimal place
    similarity_matrix = np.round(similarity_matrix, 1)

    # Plot the heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', square=True, vmin=-20, vmax=1, xticklabels=artist_names, yticklabels=artist_names, fmt=".1f")
    plt.xlabel('Artist')
    plt.ylabel('Artist')
    plt.title('Similarity between Artists')

    # Add a colorbar
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Similarity')

    # Show the plot
    plt.show()

    # Save the plot if a filename is provided
    if out_filename:
        plt.savefig(out_filename)


if __name__ == "__main__":
    gBp = nx.read_graphml("../Session_2/gBp.graphml")
    gBp = get_degree_distribution(gBp)
    plot_degree_distribution(gBp)

    gDp = nx.read_graphml("../Session_2/gDp.graphml")
    gDp = get_degree_distribution(gDp)
    plot_degree_distribution(gDp)

    gw = nx.read_graphml("../Session_2/gw.graphml")
    gw_ds = get_degree_distribution(gw)
    plot_degree_distribution(gw_ds)

    most_similar_artist = get_most_similar_artist(gw, "06HL4z0CvFAxyc27GXpf02") # the id of Taylor Swift in Spotify
    least_similar_artist = get_least_similar_artist(gw, "06HL4z0CvFAxyc27GXpf02")

    df = pd.read_csv("../Session_1/songs.csv")
    df = compute_mean_audio_features(df)
    plot_audio_features(df, artist1_id = "06HL4z0CvFAxyc27GXpf02", artist2_id = most_similar_artist) 
    plot_audio_features(df, artist1_id = "06HL4z0CvFAxyc27GXpf02", artist2_id = least_similar_artist)
    
    plot_similarity_heatmap(df, similarity = "euclidean")

    connected_components = list(nx.connected_components(gw))
    largest_component = max(connected_components, key=len)
    largest_component_graph = gw.subgraph(largest_component)

    min_percentile_range = range(0, 101, 10)  # Prune edges with weights below the 0th, 10th, ..., 100th percentile
    prune_and_plot(gw, min_percentile_range)
    
    

    
    

    