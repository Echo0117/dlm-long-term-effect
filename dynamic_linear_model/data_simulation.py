import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from config import config
import dynamic_linear_model.utils as utils

class DataSimulation:
    def __init__(self, X_t: np.ndarray, Z_t: np.ndarray, Y_t: np.ndarray, G: float = config["modelTraining"]["originalG"], theta_0: float = config["modelTraining"]["theta0"]):
        """
        Initialize the DataSimulation class with data and parameters.

        :param X_t: Independent variables matrix X.
        :param Z_t: Independent variables matrix Z.
        :param Y_t: Dependent variable vector Y.
        :param G: State transition coefficient.
        :param theta_0: Initial state.
        """
        self.X_t = X_t
        self.Z_t = Z_t
        self.Y_t = Y_t
        self.G = G
        self.theta_0 = theta_0
        self.eta = None
        self.zeta = None

    def obtain_parameters_by_lr(self):
        """
        Obtain eta (η) and zeta (ζ) by fitting a linear regression model to XZ_t and Y_t.
        """
        XZ_t = np.hstack((self.X_t, self.Z_t))
        model_XZ = LinearRegression().fit(XZ_t, self.Y_t)
        eta_zeta = model_XZ.coef_
        
        self.eta = eta_zeta[:self.X_t.shape[1]]
        self.zeta = eta_zeta[self.X_t.shape[1]:]

    def generate_theta(self) -> np.ndarray:
        """
        Generate theta values using the state transition equation.
        
        :return: Array of theta values.
        """
        T = len(self.Y_t)
        theta_t = np.zeros(T)
        theta_t[0] = self.theta_0
        
        for t in range(T):
            if t > 0:
                theta_t[t] = self.G * theta_t[t-1] + np.dot(self.Z_t[t-1], self.zeta / 2)
        
        return theta_t

    def generate_simulated_Y(self) -> np.ndarray:
        """
        Generate simulated Y values using the DLM model.
        
        :return: Array of predicted Y values.
        """
        self.obtain_parameters_by_lr()
        T = len(self.Y_t)
        theta_t = self.generate_theta()
        predicted_Y = np.zeros(T)
        
        for t in range(T):
            predicted_Y[t] = theta_t[t] + np.dot(self.X_t[t], self.eta) + np.dot(self.Z_t[t], self.zeta / 2)
        
        return predicted_Y

    def get_simulation_results(self) -> dict:
        """
        Get a DataFrame with actual and predicted Y values for visualization.
        
        :return: DataFrame containing actual and predicted Y values.
        """
        predicted_Y = self.generate_simulated_Y()
        results_Y = {
            'actual_Y': self.Y_t,
            'simulated_Y': predicted_Y
        }
        utils.log_parameters_results(self.G, self.eta, self.zeta)
        return results_Y

