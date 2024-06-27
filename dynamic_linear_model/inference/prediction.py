from dynamic_linear_model.model import DynamicLinearModel
from config import config
from dynamic_linear_model.inference.samplying_method import MCMCSamplingMethod

class SalesPrediction:
    def __init__(self):
        """
        Initialize the SalesPrediction class with DLM model.
        """
        self.optimized_params = None

        self.model = DynamicLinearModel(config["modelTraining"]["modelPath"])

    def predict(self, X_t, Z_t):
        """
        Predict the values using the optimized parameters and the Dynamic Linear Model.

        Parameters:
        initial_G (float): Initial value of G.
        initial_eta (np.ndarray): Initial values of eta.
        initial_zeta (np.ndarray): Initial values of zeta.

        Returns:
        np.ndarray: Predicted values.
        """
        # Optimize parameters
        # G, eta, zeta = self.optimize_parameters()
        # T = len(self.Y_t)
        # Use the DLM model to predict values
        predicted_Y = self.model.dlm_model(X_t, Z_t)
        return predicted_Y

    def save(self):
        """
        Save the model parameters to a file.
        """
        pass

    def load(self):
        """
        Load the model parameters from a file.
        """
        pass
