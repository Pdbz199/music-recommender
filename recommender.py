import numpy as np
import os
import random
import spotipy
import json
import pickle

from dotenv import load_dotenv
from linucb import LinUCB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

load_dotenv()

PLAYLIST_DATA_CACHE = "playlist_data.pkl"

class MusicRecommender:
    def __init__(self, playlist_name="Good Songs"):
        self.scope = (
            "user-library-read", "playlist-read-private"
        )
        self.cache_path = ".spotipyoauthcache"
        self.user_spotify_client = self._get_user_spotify_client()
        self.public_spotify_client = self._get_public_spotify_client()

        self.feature_keys = self._load_genres() + ['track_popularity', 'artist_popularity', 'duration_ms']
        self.scaler = StandardScaler()

        self.playlist_name = playlist_name
        self.tracks_data = {}  # Using a dictionary for easier updates
        self.song_clusters = []

        self._initialize_recommender()

    def _load_genres(self):
        with open('genres.json', 'r') as f:
            return json.load(f)

    def _initialize_recommender(self):
        # Load from cache if it exists
        if os.path.exists(PLAYLIST_DATA_CACHE):
            print("Loading cached playlist data...")
            with open(PLAYLIST_DATA_CACHE, 'rb') as f:
                self.tracks_data = pickle.load(f)
            print(f"Loaded {len(self.tracks_data)} tracks from cache.")

        # Fetch latest playlist and find the difference
        print("Fetching latest playlist state...")
        latest_tracks = self._get_playlist_tracks()
        latest_track_ids = {t['id'] for t in latest_tracks if t and t.get('id')}
        cached_track_ids = set(self.tracks_data.keys())

        added_ids = latest_track_ids - cached_track_ids
        removed_ids = cached_track_ids - latest_track_ids

        # Process new and removed tracks
        if added_ids:
            print(f"Found {len(added_ids)} new songs. Adding to cache...")
            new_tracks = [t for t in latest_tracks if t['id'] in added_ids]
            new_features = self.get_track_features(new_tracks)
            for track, features in zip(new_tracks, new_features):
                if features is not None:
                    self.tracks_data[track['id']] = {'track': track, 'features': features}

        if removed_ids:
            print(f"Found {len(removed_ids)} removed songs. Removing from cache...")
            for track_id in removed_ids:
                del self.tracks_data[track_id]

        # Re-cluster if there were any changes
        if added_ids or removed_ids or not self.song_clusters:
            print("Change in playlist detected or no clusters found. Re-clustering...")
            self._create_song_clusters()

        # Save the updated cache
        print("Saving updated cache...")
        with open(PLAYLIST_DATA_CACHE, 'wb') as f:
            pickle.dump(self.tracks_data, f)
        print("Cache updated.")

    def _get_playlist_tracks(self):
        # ... implementation to find the playlist by name and fetch all tracks
        playlists = self.user_spotify_client.current_user_playlists()
        playlist_id = None
        for playlist in playlists['items']:
            if playlist['name'] == self.playlist_name:
                playlist_id = playlist['id']
                break

        if not playlist_id:
            print(f"Playlist '{self.playlist_name}' not found.")
            return []

        all_tracks = []
        results = self.user_spotify_client.playlist_items(playlist_id)
        while results:
            all_tracks.extend([item['track'] for item in results['items'] if item['track']])
            if results['next']:
                results = self.user_spotify_client.next(results)
            else:
                results = None
        return all_tracks

    def get_recommendations(self, seed_tracks=None, limit=20):
        # This is a replacement for the deprecated recommendations endpoint.
        # It generates candidates by finding playlists related to the seed tracks' genres.
        if not seed_tracks:
            return []

        saved_track_ids = {track['id'] for track in self.playlist_tracks if track and track.get('id')}

        # 1. Get genres from seed tracks
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

        # 2. For a sample of genres, find related playlists
        candidate_tracks = []
        genres_to_search = random.sample(list(seed_genres), min(len(seed_genres), 5))

        for genre in genres_to_search:
            try:
                # Format search query as requested: genre:"Deep Discofox"
                search_query = f'genre:"{genre}"'
                search_results = self.public_spotify_client.search(q=search_query, type='playlist', limit=10)

                # Add a check to ensure playlists were found
                if not search_results or not search_results.get('playlists') or not search_results['playlists'].get('items'):
                    print(f"No playlists found for genre '{genre}'. Skipping.")
                    continue

                playlists = search_results['playlists']['items']
                if not playlists:
                    continue

                playlist = random.choice(playlists)
                playlist_tracks_response = self.public_spotify_client.playlist_items(playlist['id'], limit=50)

                if playlist_tracks_response:
                    for item in playlist_tracks_response['items']:
                        track = item.get('track')
                        if track and track.get('id') and not track.get('is_local') and track.get('id') not in saved_track_ids:
                            candidate_tracks.append(track)
            except Exception:
                # Silently skip genres that cause errors.
                continue

        # 3. Return a shuffled list of unique candidate tracks
        if not candidate_tracks:
            print("Could not find any new tracks from playlists.")
            return []

        unique_tracks = list({t['id']: t for t in candidate_tracks if t and t.get('id')}.values())
        random.shuffle(unique_tracks)
        return unique_tracks[:limit]


    def get_track_features(self, tracks):
        features_list = []
        for track in tracks:
            if not track or not track.get('id') or track.get('is_local'):
                features_list.append(None)
                continue

            try:
                artist_id = track['artists'][0]['id']
                artist = self.public_spotify_client.artist(artist_id)

                genre_features = [1 if genre in artist['genres'] else 0 for genre in self._load_genres()]

                popularity_features = [
                    track['popularity'] / 100.0,
                    artist['popularity'] / 100.0,
                    track['duration_ms'] / 360000.0,  # Normalize duration
                ]

                features_list.append(np.array(genre_features + popularity_features))
            except Exception as e:
                # Be less verbose for individual track errors
                # print(f"Could not fetch features for track {track['id']}: {e}")
                features_list.append(None)
        return features_list

    def _create_song_clusters(self, n_clusters=50): # Increased clusters for larger playlist
        if not self.tracks_data:
            return

        tracks_with_features = list(self.tracks_data.values())
        tracks = [d['track'] for d in tracks_with_features]
        feature_matrix = np.array([d['features'] for d in tracks_with_features])

        if len(tracks) < n_clusters:
            n_clusters = len(tracks)

        self.scaler.fit(feature_matrix)
        scaled_features = self.scaler.transform(feature_matrix)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)

        self.song_clusters = [[] for _ in range(n_clusters)]
        for i, track in enumerate(tracks):
            label = labels[i]
            self.tracks_data[track['id']]['cluster_label'] = label
            self.song_clusters[label].append(track)

        self.song_clusters = [cluster for cluster in self.song_clusters if cluster]
        if self.song_clusters:
            print(f"Successfully created {len(self.song_clusters)} clusters from your music library.")

    def _get_public_spotify_client(self):
        auth_manager = SpotifyClientCredentials()
        return spotipy.Spotify(auth_manager=auth_manager)

    def _get_user_spotify_client(self):
        spotipy_oauth = SpotifyOAuth(
            scope=self.scope,
            cache_path=self.cache_path,
            open_browser=False,
        )
        return spotipy.Spotify(auth_manager=spotipy_oauth)
