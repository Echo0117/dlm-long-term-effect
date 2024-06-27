import logging
import numpy as np
from config import config
import dynamic_linear_model.utils as utils

class GradientDescent:
    def __init__(self, learning_rate=config["modelTraining"]["learningRate"]):
        """
        Initialize the GradientDescent class with a learning rate.

        Parameters:
        learning_rate (float): The learning rate for the gradient descent optimization.
        """
        self.learning_rate = learning_rate

    def update(self):
        """
        Update the parameters. Placeholder method to be implemented in subclasses.
        """
        pass

    def step(self):
        """
        Perform a single optimization step. Placeholder method to be implemented in subclasses.
        """
        pass

    def optimize(self):
        """
        Optimize the parameters. Placeholder method to be implemented in subclasses.
        """
        pass

class GradientDescentPerturbation(GradientDescent):
    def __init__(self, learning_rate=config["modelTraining"]["learningRate"], n_iterations=config["modelTraining"]["epoch"]):
        """
        Initialize the GradientDescentPerturbation class with a learning rate and number of iterations.

        Parameters:
        learning_rate (float): The learning rate for the gradient descent optimization.
        n_iterations (int): The number of iterations for the optimization.
        """
        super().__init__(learning_rate)
        self.n_iterations = n_iterations

    def optimize(self, Y_t, X_t, Z_t, initial_params):
        """
        Optimize the parameters using perturbation-based gradient descent.

        Parameters:
        Y_t (np.ndarray): Dependent variable vector Y.
        X_t (np.ndarray): Independent variables matrix X.
        Z_t (np.ndarray): Independent variables matrix Z.
        initial_params (list): List of initial parameters (G, eta, zeta).

        Returns:
        np.ndarray: Optimized parameters.
        """
        params = np.array(initial_params)
        n_params = len(params)
        
        for iteration in range(self.n_iterations):
            gradients = np.zeros(n_params)
            
            for i in range(n_params):
                original_param = params[i]
                loss_original = utils.negative_log_likelihood(params, Y_t, X_t, Z_t)
                params[i] = original_param + 1e-5
                loss_perturbed = utils.negative_log_likelihood(params, Y_t, X_t, Z_t)
                gradients[i] = (loss_perturbed - loss_original) / 1e-5
                params[i] = original_param
            
            params -= self.learning_rate * gradients
            
            if iteration % 100 == 0:
                logging.info(f"Iteration {iteration}: Loss = {loss_original}")
                if np.isnan(loss_original):
                    logging.info("NaN encountered in loss. Exiting.")
                    break
        
        return params
