import numpy as np
import pickle

class LinUCB:
    def __init__(self, n_dims, alpha=1.0):
        self.n_dims = n_dims
        self.alpha = alpha
        self.A = np.identity(n_dims)
        self.b = np.zeros((n_dims, 1))

    def predict(self, context):
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        p = theta.T @ context + self.alpha * np.sqrt(context.T @ A_inv @ context)
        return p

    def update(self, context, reward):
        self.A += context @ context.T
        self.b += reward * context

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
