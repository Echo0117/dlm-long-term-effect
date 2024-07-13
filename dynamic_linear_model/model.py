import os
import pickle
import numpy as np


class DynamicLinearModel:
    def __init__(self, model_path=None):
        """
        Initialize the Dynamic Linear Model class.

        Parameters:
        model_path (str): Path to the saved model parameters.
        X_t (np.ndarray): Independent variables matrix X (required if model_path is None).
        Z_t (np.ndarray): Independent variables matrix Z (required if model_path is None).
        """
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        # else:
        #     self.G = np.random.normal(0, 1)
        #     self.eta = np.random.normal(size=X_t.shape[1])
        #     self.zeta = np.random.normal(size=Z_t.shape[1])
        #     self.T = X_t.shape[0]

    def _load_model(self, model_path):
        """
        Load the model parameters from a file.

        Parameters:
        model_path (str): Path to the saved model parameters.
        """
        with open(model_path, "rb") as file:
            model_data = pickle.load(file)
            self.G = model_data["G"]
            self.eta = model_data["eta"]
            self.zeta = model_data["zeta"]

    def dlm_model(self, X_t, Z_t):
        """
        Apply the Dynamic Linear Model (DLM) to predict values.

        Parameters:
        X_t (np.ndarray): Independent variables matrix X.
        Z_t (np.ndarray): Independent variables matrix Z.
        eta (np.ndarray): Coefficients for X_t.
        zeta (np.ndarray): Coefficients for Z_t.
        G (float): State transition coefficient.
        T (int): Number of time steps.

        Returns:
        np.ndarray: Predicted values.
        """
        T = len(X_t)
        theta_t = np.zeros(T)
        predicted_Y = np.zeros(T)

        for t in range(T):
            if t > 0:
                # State transition equation
                theta_t[t] = self.G * theta_t[t - 1] + np.dot(Z_t[t - 1], self.zeta / 2)

            # Observation equation
            predicted_Y[t] = (
                theta_t[t] + np.dot(X_t[t], self.eta) + np.dot(Z_t[t], self.zeta / 2)
            )

        return predicted_Y
