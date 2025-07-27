import numpy as np
import pandas as pd
import pickle
import plotly.express as px

from datetime import datetime
from sklearn.manifold import TSNE

PLAYLIST_DATA_CACHE = "playlist_data.pkl"
RECOMMENDED_TRACKS_LOG = "recommended_tracks.pkl"

# Default timestamp assigned to historical entries without a timestamp
DEFAULT_CREATED_AT = "2025-07-27T10:00:00"

def visualize_clusters():
    """
    Generates an interactive HTML visualization of the song clusters and new recommendations.
    """

    # Load the pre-processed data cached by the recommender
    print("Loading cached playlist data for visualization...")
    with open(PLAYLIST_DATA_CACHE, 'rb') as f:
        viz_data = pickle.load(f)['viz_data']

    # Load the newly recommended tracks and their features
    recommended_data = []
    try:
        with open(RECOMMENDED_TRACKS_LOG, 'rb') as f:
            recommended_data = pickle.load(f)
    except FileNotFoundError:
        print("Recommendation log not found. Only playlist songs will be plotted.")

    # Ensure every recommendation has a 'created_at' field
    for rec in recommended_data:
        if 'created_at' not in rec:
            rec['created_at'] = DEFAULT_CREATED_AT

    # Combine playlist data and recommendation data for a unified visualization
    all_tracks = viz_data['tracks'] + [rec['track'] for rec in recommended_data]
    all_features = (
        np.vstack([viz_data['reduced_features'], np.array([rec['features'] for rec in recommended_data])])
        if recommended_data else viz_data['reduced_features']
    )

    # Create labels for plotting
    playlist_labels = [str(c) for c in viz_data['cluster_labels']]
    recommendation_labels = ['New Recommendation'] * len(recommended_data)
    all_labels = playlist_labels + recommendation_labels

    # Prepare the data into a Pandas DataFrame
    df = pd.DataFrame(all_tracks)
    df['cluster'] = all_labels
    df['text'] = df['name'] + ' by ' + df['artists'].apply(lambda artists: artists[0]['name'])
    df['is_recommended'] = df['cluster'] == 'New Recommendation'

    # Identify the most recent recommendation (if any) and flag it
    most_recent_rec = None
    if recommended_data:
        most_recent_rec = max(recommended_data, key=lambda r: r['created_at'])

    def is_most_recent(row):
        if not most_recent_rec:
            return False
        return row['id'] == most_recent_rec['track']['id']

    df['is_most_recent'] = df.apply(is_most_recent, axis=1)

    # Perform t-SNE on the combined feature set
    print("Performing t-SNE dimensionality reduction on all tracks...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    # Separate most-recent point to avoid double-plotting (after coordinates added)
    df_non_recent = df[~df['is_most_recent']].copy()
    df_recent = df[df['is_most_recent']].copy()

    # Create an interactive scatter plot
    print("Generating interactive plot...")
    fig = px.scatter(
        df_non_recent,
        x="tsne-2d-one",
        y="tsne-2d-two",
        color="cluster",
        hover_name="text",
        size_max=15,
        symbol='is_recommended',
        symbol_map={True: 'star', False: 'circle'},
        title="Song Clusters and New Recommendations",
    )

    # Enlarge all star (recommended) markers for better visibility
    for trace in fig.data:
        symbol = getattr(trace.marker, 'symbol', None)
        if symbol == 'star':
            trace.update(marker=dict(size=15))

    # Add the most recent recommendation as a standalone red star with detailed hover
    if not df_recent.empty:
        fig.add_scatter(
            x=df_recent['tsne-2d-one'],
            y=df_recent['tsne-2d-two'],
            mode='markers',
            marker=dict(
                symbol='star',
                size=22,
                color='red',
                line=dict(color='black', width=2)
            ),
            name='Most Recent Recommendation',
            customdata=df_recent[['cluster', 'is_recommended']].values,
            text=df_recent['text'],
            hovertemplate=(
                '%{text}<br>'
                'cluster=%{customdata[0]}<br>'
                'is_recommended=%{customdata[1]}<br>'
                'tsne-2d-one=%{x:.6f}<br>'
                'tsne-2d-two=%{y:.6f}<extra></extra>'
            )
        )

    # Save the generated plot to an HTML file
    fig.write_html("song_clusters.html")
    print("\nSuccess! Interactive plot saved to song_clusters.html")

if __name__ == "__main__":
    visualize_clusters()
