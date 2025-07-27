import json
import numpy as np
import os
import pickle
import random
import spotipy
import time

from dotenv import load_dotenv
from linucb import LinUCB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

load_dotenv()

PLAYLIST_DATA_CACHE = "playlist_data.pkl"

class MusicRecommender:
    """
    A music recommendation system that uses a LinUCB bandit model to learn a user's taste.

    This class handles all interactions with the Spotify API, processes song data into
    feature vectors, clusters songs to understand a user's taste profile, and generates
    recommendations by exploring public playlists. It also caches processed data to
    ensure fast startup times on subsequent runs.
    """

    def __init__(self, playlist_name="Good Songs"):
        """
        Initializes the MusicRecommender.

        Args:
            playlist_name (str): The name of the user's Spotify playlist to use as the seed for their taste.
        """

        self.scope = (
            "user-library-read",
            "playlist-read-private",
        )
        self.cache_path = ".spotipyoauthcache"
        self.user_spotify_client = self._get_user_spotify_client()
        self.public_spotify_client = self._get_public_spotify_client()

        self.playlist_name = playlist_name
        self.tracks_data = {}  # Using a dictionary for easier updates: {track_id: {'track': {...}, 'features': [...]}}
        self.song_clusters = []
        self.data_reprocessed = False # This flag will track if we re-processed the data.

        # Load and process genres once for efficiency
        loaded_genres = self._load_genres()
        self.genre_names = [genre['name'].lower() for genre in loaded_genres]
        self.feature_keys = self.genre_names + ['track_popularity', 'artist_popularity', 'duration_ms']
        self.scaler = StandardScaler()
        self.pca = None  # Will hold the fitted PCA model

        self._initialize_recommender()

    def _load_genres(self):
        """Loads the comprehensive genre list from the local JSON file."""
        with open('genres.json', 'r') as f:
            return json.load(f)

    def _initialize_recommender(self):
        """
        Initializes the recommender by loading cached data or fetching and processing it from Spotify.

        This method implements an incremental update strategy. It loads the last known state
        from a cache file, fetches the current state of the user's playlist, and only
        processes the songs that have been added or removed, saving significant time.
        """

        # Load the set of all track IDs from the last run, if the cache exists
        cached_track_ids = set()
        if os.path.exists(PLAYLIST_DATA_CACHE):
            print("Loading cached playlist data...")
            with open(PLAYLIST_DATA_CACHE, 'rb') as f:
                cached_data = pickle.load(f)
                self.tracks_data = cached_data.get('tracks_data', {})
                cached_track_ids = cached_data.get('all_track_ids', set())
            print(f"Loaded {len(self.tracks_data)} processed tracks from cache.")

        # Fetch the current state of the playlist from Spotify
        print("Fetching latest playlist state...")
        latest_tracks = self._get_playlist_tracks()
        latest_track_ids = {t['id'] for t in latest_tracks if t and t.get('id')}

        # Compare the live playlist with the cached set of IDs to detect real changes
        added_ids = latest_track_ids - cached_track_ids
        removed_ids = cached_track_ids - latest_track_ids

        # If the playlist has actually changed, or if there's no cache, we re-process everything
        if added_ids or removed_ids or not os.path.exists(PLAYLIST_DATA_CACHE):
            self.data_reprocessed = True
            print("Change in playlist detected or no cache found. Processing all tracks...")

            # Rebuild the tracks_data dictionary from scratch
            all_features = self.get_track_features(latest_tracks)
            self.tracks_data = {
                track['id']: {'track': track, 'features': features}
                for track, features in zip(latest_tracks, all_features) if features is not None
            }

            self._create_song_clusters()

            print("Saving updated cache...")
            with open(PLAYLIST_DATA_CACHE, 'wb') as f:
                pickle.dump({
                    'tracks_data': self.tracks_data,
                    'all_track_ids': latest_track_ids,
                    'scaler': self.scaler,
                    'pca': self.pca,
                    'viz_data': self.tracks_data_for_viz,
                }, f)
            print("Cache updated.")
        else:
            # If no changes, load the rest of the data from the cache and proceed
            with open(PLAYLIST_DATA_CACHE, 'rb') as f:
                cached_data = pickle.load(f)
            self.scaler = cached_data.get('scaler', self.scaler)
            self.pca = cached_data.get('pca', self.pca)
            self._create_song_clusters()

    def _spotify_api_call(self, api_callable):
        """
        A robust wrapper for making calls to the Spotify API.

        This method implements a retry mechanism with exponential backoff and jitter
        to gracefully handle rate-limiting errors (HTTP 429), making the application
        more resilient.

        Args:
            api_callable (function): A lambda function that encapsulates the Spotipy API call.

        Returns:
            The result of the API call, or None if it fails after multiple retries.
        """

        max_retries = 5
        base_wait_time = 5  # Start with 5 seconds
        for attempt in range(max_retries):
            try:
                return api_callable()
            except spotipy.SpotifyException as e:
                if e.http_status == 429:  # Rate limit error
                    # Respect the 'Retry-After' header if provided by the API
                    wait_time = int(e.headers.get('Retry-After', base_wait_time * (2 ** attempt)))
                    print(f"Rate limited by API. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time + random.uniform(0, 1))  # Add jitter to avoid thundering herd problem
                else:
                    # For other errors (403, 404, etc.), don't retry
                    print(f"An unrecoverable Spotify API error occurred: {e}")
                    return None
        print("API call failed after max retries.")
        return None

    def _get_playlist_tracks(self):
        """
        Fetches all tracks from the user's specified playlist, handling pagination for both
        the list of playlists and the tracks within the target playlist.

        Returns:
            list: A list of track objects from the playlist.
        """

        # Paginate through all user playlists to find the correct one by name
        playlist_id = None
        playlists_response = self._spotify_api_call(lambda: self.user_spotify_client.current_user_playlists(limit=50))
        while playlists_response:
            for playlist in playlists_response['items']:
                if playlist['name'] == self.playlist_name:
                    playlist_id = playlist['id']
                    break
            if playlist_id:
                break
            if playlists_response['next']:
                playlists_response = self._spotify_api_call(lambda: self.user_spotify_client.next(playlists_response))
            else:
                playlists_response = None

        if not playlist_id:
            print(f"Playlist '{self.playlist_name}' not found.")
            return []

        # Fetch all tracks from the playlist, page by page, with logging
        all_tracks = []
        page = 1
        results = self._spotify_api_call(lambda: self.user_spotify_client.playlist_items(playlist_id, limit=100))

        while results:
            print(f"Fetched page {page} with {len(results['items'])} tracks...")
            all_tracks.extend([item['track'] for item in results['items'] if item['track']])
            page += 1
            if results['next']:
                results = self._spotify_api_call(lambda: self.user_spotify_client.next(results))
            else:
                results = None
        return all_tracks

    def get_recommendations(self, seed_tracks=None, limit=20):
        """
        Generates a list of candidate tracks for recommendation.

        This method serves as a replacement for the deprecated `/recommendations` endpoint.
        It generates candidates by finding public playlists that are related to the genres
        of the seed tracks, effectively creating our own recommendation pool.

        Args:
            seed_tracks (list): A list of track IDs to use as seeds.
            limit (int): The maximum number of recommended tracks to return.

        Returns:
            list: A list of new, unheard track objects to be considered for recommendation.
        """

        # It generates candidates by finding playlists related to the seed tracks' genres
        if not seed_tracks:
            return []

        # Correctly get the set of saved track IDs from the keys of the tracks_data dictionary
        saved_track_ids = set(self.tracks_data.keys())

        # Determine the genres of the seed tracks by looking up their artists
        seed_genres = set()
        try:
            tracks_info = self.public_spotify_client.tracks(seed_tracks)
            for track in tracks_info['tracks']:
                if track and track.get('artists'):
                    artist_id = track['artists'][0]['id']
                    artist = self.public_spotify_client.artist(artist_id)
                    seed_genres.update(artist['genres'])
        except Exception as e:
            print(f"Error fetching seed track genres: {e}")
            return []

        if not seed_genres:
            print("Could not find genres for seed tracks. Using default genres.")
            seed_genres = set(self.feature_keys[:5])

        # Search for public playlists that match a sample of the seed genres
        candidate_tracks = []
        genres_to_search = random.sample(list(seed_genres), min(len(seed_genres), 5))

        for genre in genres_to_search:
            try:
                # Format search query correctly, e.g., genre:"Deep Discofox"
                search_query = f'genre:"{genre}"'
                search_results = self.public_spotify_client.search(q=search_query, type='playlist', limit=10)

                # Add a check to ensure playlists were found
                if not search_results or not search_results.get('playlists') or not search_results['playlists'].get('items'):
                    print(f"No playlists found for genre '{genre}'. Skipping.")
                    continue

                playlists = search_results['playlists']['items']
                if not playlists:
                    continue

                # Grab tracks from a random playlist in the search results
                playlist = random.choice(playlists)
                playlist_tracks_response = self.public_spotify_client.playlist_items(playlist['id'], limit=50)

                if playlist_tracks_response:
                    for item in playlist_tracks_response['items']:
                        track = item.get('track')
                        # Add the track if it's valid and not already in the user's library
                        if track and track.get('id') and not track.get('is_local') and track.get('id') not in saved_track_ids:
                            candidate_tracks.append(track)
            except Exception:
                # Silently skip genres that cause errors
                continue

        # Return a shuffled, unique list of candidate tracks
        if not candidate_tracks:
            print("Could not find any new tracks from playlists.")
            return []

        unique_tracks = list({t['id']: t for t in candidate_tracks if t and t.get('id')}.values())
        random.shuffle(unique_tracks)
        return unique_tracks[:limit]

    def get_track_features(self, tracks):
        """
        Converts a list of Spotify track objects into numerical feature vectors for the model.

        This is a critical step that transforms a song into a "fingerprint" the bandit can understand.
        The feature vector is a combination of:
        1. Genre Profile: A large one-hot encoded vector based on our comprehensive genre list. A '1'
           at a position means the artist is associated with that genre.
        2. Popularity Metrics: The popularity of the track and its primary artist, normalized to be
           between 0.0 and 1.0.
        3. Duration: The length of the song in milliseconds, normalized.

        Args:
            tracks (list): A list of Spotify track objects.

        Returns:
            list: A list of NumPy arrays, where each array is a feature vector for a track.
        """

        # Collect unique artist IDs from the tracks to minimize redundant API calls
        artist_ids = set()
        for t in tracks:
            if t and t.get('artists'):
                # A track can sometimes have an empty list of artists
                if t['artists']:
                    artist_id = t['artists'][0].get('id')
                    if artist_id:  # Ensure the ID is not None
                        artist_ids.add(artist_id)
        artist_ids = list(artist_ids)

        # Efficiently fetch data for all required artists in batches of 50
        artist_data_map = {}
        for i in range(0, len(artist_ids), 50):
            batch_ids = artist_ids[i:i+50]
            results = self._spotify_api_call(lambda: self.public_spotify_client.artists(batch_ids))

            if not results or not results.get('artists'):
                continue

            for artist in results['artists']:
                if artist:
                    artist_data_map[artist['id']] = {
                        'genres': artist['genres'],
                        'popularity': artist['popularity'],
                    }

        # Build the feature vector for each track using the pre-fetched artist data
        features_list = []
        for track in tracks:
            if not track or not track.get('id') or track.get('is_local'):
                features_list.append(None)
                continue

            try:
                artist_id = track['artists'][0]['id']
                artist = artist_data_map.get(artist_id)

                if not artist:
                    features_list.append(None)
                    continue

                # Use the pre-processed self.genre_names list directly
                genre_features = [1 if genre in artist['genres'] else 0 for genre in self.genre_names]

                popularity_features = [
                    track['popularity'] / 100.0,
                    artist['popularity'] / 100.0,
                    track['duration_ms'] / 360000.0,  # Normalize duration (approx. max song length)
                ]

                features_list.append(np.array(genre_features + popularity_features))
            except (KeyError, IndexError):
                features_list.append(None)

        return features_list

    def _create_song_clusters(self, n_clusters=50, n_pca_components=50):
        """
        Groups the user's songs into clusters based on their features.

        This method uses the K-Means algorithm to identify distinct "taste clusters"
        within the user's playlist, which are then used for generating diverse
        recommendation seeds.

        Args:
            n_clusters (int): The number of clusters to create.
        """

        if not self.tracks_data:
            return

        tracks_with_features = list(self.tracks_data.values())
        tracks = [d['track'] for d in tracks_with_features]
        feature_matrix = np.array([d['features'] for d in tracks_with_features])

        # PCA Step
        if feature_matrix.shape[1] > n_pca_components:
            print(f"Performing PCA to reduce dimensionality to {n_pca_components} components...")
            pca = PCA(n_components=n_pca_components)
            reduced_features = pca.fit_transform(feature_matrix)
            self.pca = pca # Save the fitted model
        else:
            reduced_features = feature_matrix
            self.pca = None

        if len(tracks) < n_clusters:
            n_clusters = len(tracks)

        # Scale features for better performance with distance-based clustering
        self.scaler.fit(reduced_features)
        scaled_features = self.scaler.transform(reduced_features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)

        # Assign cluster labels back to the main data structure
        self.song_clusters = [[] for _ in range(n_clusters)]
        for i, track in enumerate(tracks):
            label = labels[i]
            self.tracks_data[track['id']]['cluster_label'] = label
            self.song_clusters[label].append(track)

        self.song_clusters = [cluster for cluster in self.song_clusters if cluster]
        if self.song_clusters:
            print(f"Successfully created {len(self.song_clusters)} clusters from your music library.")

        # Store the reduced features and PCA model for the visualizer
        self.tracks_data_for_viz = {
            'tracks': tracks,
            'reduced_features': reduced_features,
            'cluster_labels': labels,
        }

    def _get_public_spotify_client(self):
        """Initializes a Spotipy client for accessing public data (Client Credentials Flow)."""
        auth_manager = SpotifyClientCredentials()
        return spotipy.Spotify(auth_manager=auth_manager)

    def _get_user_spotify_client(self):
        """Initializes a Spotipy client for accessing user-private data (Authorization Code Flow)."""
        spotipy_oauth = SpotifyOAuth(
            scope=self.scope,
            cache_path=self.cache_path,
            open_browser=False,
        )
        return spotipy.Spotify(auth_manager=spotipy_oauth)
