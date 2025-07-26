import pickle
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
import numpy as np

PLAYLIST_DATA_CACHE = "playlist_data.pkl"
RECOMMENDED_TRACKS_LOG = "recommended_tracks.log"

def visualize_clusters():
    # Load cached data
    with open(PLAYLIST_DATA_CACHE, 'rb') as f:
        tracks_data = pickle.load(f)

    # Load recommended tracks
    recommended_ids = set()
    try:
        with open(RECOMMENDED_TRACKS_LOG, 'r') as f:
            recommended_ids = {line.strip() for line in f}
    except FileNotFoundError:
        pass

    # Prepare data for plotting
    tracks_with_features = list(tracks_data.values())
    df = pd.DataFrame([d['track'] for d in tracks_with_features])
    feature_matrix = np.array([d['features'] for d in tracks_with_features])
    cluster_labels = [d.get('cluster_label', 'N/A') for d in tracks_with_features] # Get labels if they exist

    df['cluster'] = [str(c) for c in cluster_labels]
    df['text'] = df['name'] + ' by ' + df['artists'].apply(lambda artists: artists[0]['name'])
    df['is_recommended'] = df['id'].isin(recommended_ids)

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(feature_matrix)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    # Create plot
    fig = px.scatter(
        df,
        x="tsne-2d-one", y="tsne-2d-two",
        color="cluster",
        hover_name="text",
        size_max=15,
        symbol='is_recommended',
        symbol_map={True: 'star', False: 'circle'},
        title="Song Clusters from 'Good Songs' Playlist"
    )

    # Save to HTML
    fig.write_html("song_clusters.html")
    print("Interactive plot saved to song_clusters.html")

if __name__ == "__main__":
    visualize_clusters()