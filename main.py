import os
import random
import numpy as np
from recommender import MusicRecommender
from linucb import LinUCB

MODEL_PATH = "linucb_model.pkl"

RECOMMENDED_TRACKS_LOG = "recommended_tracks.log"

def log_recommendation(track_id):
    with open(RECOMMENDED_TRACKS_LOG, 'a') as f:
        f.write(f"{track_id}\n")

def main():
    recommender = MusicRecommender(playlist_name="Good Songs")

    if not recommender.playlist_tracks or not recommender.song_clusters:
        print("Exiting: Not enough data to proceed with recommendations.")
        return

    # Load or initialize LinUCB bandit
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        bandit = LinUCB.load_model(MODEL_PATH)
    else:
        print("Initializing new model...")
        n_dims = len(recommender.feature_keys)
        bandit = LinUCB(n_dims=n_dims, alpha=1.0)

    try:
        # Recommendation Loop
        for i in range(10):
            print(f"\n--- Round {i+1} ---")

            # 1. Generate seeds from clusters
            seed_tracks_ids = []
            if len(recommender.song_clusters) >= 5:
                selected_clusters = random.sample(recommender.song_clusters, 5)
            else:
                selected_clusters = recommender.song_clusters

            for cluster in selected_clusters:
                if cluster:
                    seed_tracks_ids.append(random.choice(cluster)['id'])

            if not seed_tracks_ids:
                print("Could not generate seeds from clusters. Using random tracks from playlist.")
                seed_tracks_ids = [track['id'] for track in random.sample(recommender.playlist_tracks, 5) if track]

            # 2. Get recommendations and their features
            try:
                recommended_tracks = recommender.get_recommendations(seed_tracks=seed_tracks_ids, limit=20)
                if not recommended_tracks:
                    print("Could not get any recommendations. Skipping round.")
                    continue

                feature_vectors = recommender.get_track_features(recommended_tracks)
            except Exception as e:
                print(f"Error getting recommendations or features: {e}")
                continue

            valid_candidates_and_features = [
                (track, features) for track, features in zip(recommended_tracks, feature_vectors) if features is not None
            ]

            if not valid_candidates_and_features:
                print("Could not generate features for any recommended tracks. Skipping round.")
                continue

            valid_candidates, feature_list = zip(*valid_candidates_and_features)

            # Scale the features
            feature_matrix = np.array(feature_list)
            scaled_features = recommender.scaler.transform(feature_matrix)

            # 3. Score and select the best track
            scores = [bandit.predict(f.reshape(-1, 1)) for f in scaled_features]
            best_track_index = np.argmax(scores)
            best_track = valid_candidates[best_track_index]
            log_recommendation(best_track['id']) # Log the recommended track

            print(f"Recommended for you: {best_track['name']} by {best_track['artists'][0]['name']}")

            # 4. Get feedback and update model
            feedback = input("Did you like this song? (y/n): ")
            reward = 1 if feedback.lower().strip() == 'y' else 0

            chosen_track_features = scaled_features[best_track_index].reshape(-1, 1)
            bandit.update(chosen_track_features, reward)
            print("Thanks! Your feedback has been recorded.")

    finally:
        # Save the model state before exiting
        print("\nSaving model state...")
        bandit.save_model(MODEL_PATH)
        print("Model saved successfully.")

if __name__ == "__main__":
    main()
