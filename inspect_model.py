import numpy as np

from linucb import LinUCB

def inspect_model(model_path="linucb_model.pkl"):
    """
    Loads a saved LinUCB model and prints its learned parameters.

    Args:
        model_path (str): The path to the saved model file.
    """

    try:
        bandit = LinUCB.load_model(model_path)
        print("Successfully loaded model from:", model_path)

        # Extract parameters
        alpha = bandit.alpha
        A = bandit.A
        b = bandit.b

        # Calculate theta
        try:
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
        except np.linalg.LinAlgError:
            print("Matrix A is singular and cannot be inverted.")
            theta = None

        print("\n--- LinUCB Model Parameters ---")
        print(f"Alpha (Exploration Parameter): {alpha}")
        print("\nMatrix A (Accumulated Information):")
        print(A)
        print("\nVector b (Accumulated Rewards):")
        print(b)

        if theta is not None:
            print("\nTheta (Coefficient Vector):")
            print(theta)
        print("\n--- End of Parameters ---")

    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_model()
