import pickle
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import numpy as np

PLAYLIST_DATA_CACHE = "playlist_data.pkl"
RECOMMENDED_TRACKS_LOG = "recommended_tracks.log"

def visualize_clusters():
    """
    Generates an interactive HTML visualization of the song clusters.

    This function loads the cached playlist data, performs t-SNE dimensionality
    reduction on the song features to create a 2D representation, and then
    plots the songs using Plotly. The resulting plot is saved as an HTML file,
    where each point represents a song, colored by its cluster, and previously
    recommended songs are marked with a star.
    """

    # Load the pre-processed data cached by the recommender
    print("Loading cached playlist data for visualization...")
    with open(PLAYLIST_DATA_CACHE, 'rb') as f:
        tracks_data = pickle.load(f)

    # Load the IDs of songs that have been recommended in previous sessions
    recommended_ids = set()
    try:
        with open(RECOMMENDED_TRACKS_LOG, 'r') as f:
            recommended_ids = {line.strip() for line in f}
    except FileNotFoundError:
        print("Recommendation log not found. No tracks will be highlighted.")
        pass

    # Prepare the data into a Pandas DataFrame for easy manipulation and plotting
    print("Preparing data for plotting...")
    tracks_with_features = list(tracks_data.values())
    df = pd.DataFrame([d['track'] for d in tracks_with_features])
    feature_matrix = np.array([d['features'] for d in tracks_with_features])
    cluster_labels = [d.get('cluster_label', 'N/A') for d in tracks_with_features]

    # Add computed columns for plotting: cluster, hover text, and recommendation status
    df['cluster'] = [str(c) for c in cluster_labels]
    df['text'] = df['name'] + ' by ' + df['artists'].apply(lambda artists: artists[0]['name'])
    df['is_recommended'] = df['id'].isin(recommended_ids)

    # Perform t-SNE dimensionality reduction to visualize the high-dimensional
    # feature vectors in 2D space
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(feature_matrix)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    # Create an interactive scatter plot using Plotly Express
    print("Generating interactive plot...")
    fig = px.scatter(
        df,
        x="tsne-2d-one",
        y="tsne-2d-two",
        color="cluster",
        hover_name="text",  # Show song name and artist on hover
        size_max=15,
        symbol='is_recommended',
        symbol_map={True: 'star', False: 'circle'},  # Use stars for recommended songs
        title="Song Clusters from 'Good Songs' Playlist",
    )

    # Save the generated plot to an HTML file
    fig.write_html("song_clusters.html")
    print("\nSuccess! Interactive plot saved to song_clusters.html")
    print("Open this file in your browser to explore your music clusters.")

if __name__ == "__main__":
    visualize_clusters()
