import numpy as np
import pickle

class LinUCB:
    """
    Implementation of the LinUCB (Linear Upper Confidence Bound) bandit algorithm.

    LinUCB is a contextual bandit algorithm that models the expected reward of an arm
    (in our case, a song) as a linear function of its context (its feature vector).
    It uses the UCB principle to balance exploration (trying new songs to learn more)
    and exploitation (recommending songs it's confident you'll like).

    Attributes:
        n_dims (int): The dimensionality of the feature vectors (context).
        alpha (float): A parameter that controls the level of exploration. Higher values
                       encourage more exploration of uncertain options.
        A (np.ndarray): A matrix that accumulates information about the features of the
                        arms that have been played. It's used to estimate the uncertainty.
        b (np.ndarray): A vector that accumulates the rewards of the arms, weighted by their
                        features. It's used to estimate the expected reward.
    """

    def __init__(self, n_dims, alpha=1.0):
        """
        Initializes the LinUCB model.

        Args:
            n_dims (int): The number of dimensions in the feature vectors.
            alpha (float): The exploration-exploitation trade-off parameter.
        """

        self.n_dims = n_dims
        self.alpha = alpha
        # Initialize A as an identity matrix. This corresponds to a Ridge regression
        # regularizer and ensures that A is always invertible
        self.A = np.identity(n_dims)
        # Initialize b as a zero vector
        self.b = np.zeros((n_dims, 1))

    def predict(self, context):
        """
        Calculates the UCB score for a given arm (context).

        The score is the sum of two components:
        1. The estimated expected reward (theta.T @ context).
        2. An uncertainty term that is larger for arms we know less about.

        Args:
            context (np.ndarray): The feature vector of the arm to be scored.

        Returns:
            float: The calculated UCB score for the arm.
        """

        # Invert the A matrix. This is the most computationally expensive step
        A_inv = np.linalg.inv(self.A)
        # Estimate the coefficient vector theta from the accumulated data.
        # This is equivalent to solving a Ridge regression problem
        theta = A_inv @ self.b
        # Calculate the UCB score. The first term is the exploitation part,
        # and the second term (with alpha) is the exploration part
        p = theta.T @ context + self.alpha * np.sqrt(context.T @ A_inv @ context)
        return p

    def update(self, context, reward):
        """
        Updates the model's parameters with the observed reward for a chosen arm.

        Args:
            context (np.ndarray): The feature vector of the arm that was chosen.
            reward (int): The observed reward (1 for a like, 0 for a dislike).
        """

        # Update the A matrix with the features of the chosen arm
        self.A += context @ context.T
        # Update the b vector with the reward and features of the chosen arm
        self.b += reward * context

    def save_model(self, path):
        """Saves the current state of the model to a file using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        """Loads a saved model state from a file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
