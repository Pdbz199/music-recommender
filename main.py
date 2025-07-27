import numpy as np
import os
import pickle
import random

from datetime import datetime
from linucb import LinUCB
from recommender import MusicRecommender

MODEL_PATH = "linucb_model.pkl"
RECOMMENDED_TRACKS_LOG = "recommended_tracks.pkl"

def log_recommendation(track, features):
    """Logs a recommended track and its feature vector."""
    recommendations = []
    if os.path.exists(RECOMMENDED_TRACKS_LOG):
        with open(RECOMMENDED_TRACKS_LOG, 'rb') as f:
            recommendations = pickle.load(f)

    # Append the new recommendation with a timestamp for easier sorting later
    recommendations.append({
        'track': track,
        'features': features,
        'created_at': datetime.now().isoformat()
    })

    with open(RECOMMENDED_TRACKS_LOG, 'wb') as f:
        pickle.dump(recommendations, f)

def main():
    """
    The main entry point for the music recommender application.

    This function orchestrates the entire recommendation process. It initializes the
    recommender, loads or creates the bandit model, runs the interactive recommendation
    loop, and ensures the model's state is saved upon exit.
    """

    # Initialize the recommender, which handles data loading, caching, and processing
    recommender = MusicRecommender(playlist_name="Good Songs")

    # Exit if the recommender could not load or process enough data
    if not recommender.tracks_data or not recommender.song_clusters:
        print("Exiting: Not enough data to proceed with recommendations.")
        return

    # Load the bandit model if it already exists; otherwise start from scratch
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        bandit = LinUCB.load_model(MODEL_PATH)
        # Verify that the model dimensions match the data
        if bandit.n_dims != recommender.pca.n_components_:
            print("Model dimensions mismatch! Initializing a new model.")
            bandit = None
    else:
        bandit = None

    if bandit is None:
        print("Initializing new model...")
        n_dims = recommender.pca.n_components_ if recommender.pca else len(recommender.feature_keys)
        bandit = LinUCB(n_dims=n_dims, alpha=1.0)

    try:
        # Start the continuous interactive recommendation loop
        while True:
            print(f"\n--- Next Round ---")

            # Generate a diverse set of seed tracks by sampling from the taste clusters
            seed_tracks_ids = []
            if len(recommender.song_clusters) >= 5:
                selected_clusters = random.sample(recommender.song_clusters, 5)
            else:
                selected_clusters = recommender.song_clusters

            for cluster in selected_clusters:
                if cluster:
                    seed_tracks_ids.append(random.choice(cluster)['id'])

            # As a fallback, use random tracks if cluster-based seeding fails
            if not seed_tracks_ids:
                print("Could not generate seeds from clusters. Using random tracks from playlist.")
                all_tracks = [data['track'] for data in recommender.tracks_data.values()]
                seed_tracks_ids = [track['id'] for track in random.sample(all_tracks, 5) if track]

            # Get a list of new candidate songs and process their features
            try:
                recommended_tracks = recommender.get_recommendations(seed_tracks=seed_tracks_ids, limit=20)
                if not recommended_tracks:
                    print("Could not get any recommendations. Skipping round.")
                    continue

                feature_vectors = recommender.get_track_features(recommended_tracks)
            except Exception as e:
                print(f"Error getting recommendations or features: {e}")
                continue

            # Filter out any candidates for which feature generation failed
            valid_candidates_and_features = [
                (track, features) for track, features in zip(recommended_tracks, feature_vectors) if features is not None
            ]

            if not valid_candidates_and_features:
                print("Could not generate features for any recommended tracks. Skipping round.")
                continue

            # Unzip the list of valid candidates and their feature vectors
            valid_candidates, feature_list = zip(*valid_candidates_and_features)

            # Convert the list of feature vectors into a matrix
            feature_matrix = np.array(feature_list)

            # Apply the SAME PCA transformation that was fitted on the playlist data
            if recommender.pca:
                reduced_features = recommender.pca.transform(feature_matrix)
            else:
                reduced_features = feature_matrix

            # Scale the features to be used by the model
            scaled_features = recommender.scaler.transform(reduced_features)

            # Use the bandit model to score each candidate and select the best one
            scores = [bandit.predict(f.reshape(-1, 1)) for f in scaled_features]
            best_track_index = np.argmax(scores)
            best_track = valid_candidates[best_track_index]

            print(f"Recommended for you: {best_track['name']} by {best_track['artists'][0]['name']}")

            # Get user feedback and update the model
            feedback = input("Did you like this song? (y/n): ")
            reward = 1 if feedback.lower().strip() == 'y' else 0

            # Log the recommendation only after feedback is received
            recommended_features = reduced_features[best_track_index]
            log_recommendation(best_track, recommended_features)

            # Update the bandit model with the user's feedback
            chosen_track_features = scaled_features[best_track_index].reshape(-1, 1)
            bandit.update(chosen_track_features, reward)

            # Persist the updated model immediately so progress isn't lost
            bandit.save_model(MODEL_PATH)
            print("Thanks! Your feedback has been recorded and the model has been saved.")
    except KeyboardInterrupt:
        print("\nExiting and saving model state...")
    finally:
        # Ensure the model's learned state is always saved when the script exits
        print("\nSaving model state...")
        bandit.save_model(MODEL_PATH)
        print("Model saved successfully.")

if __name__ == "__main__":
    main()
