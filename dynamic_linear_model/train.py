import logging
import pickle

from matplotlib import pyplot as plt
from config import config
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dynamic_linear_model.inference.gradient_descent import GradientDescentPerturbation
import dynamic_linear_model.utils as utils
from dynamic_linear_model.model import DynamicLinearModel
from dynamic_linear_model.evaluations.metrics import Metrics
from sklearn.model_selection import KFold, train_test_split




class Trainer:
    def __init__(self, X_t, Z_t, Y_t, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
            """
            Initialize the Trainer class with the given parameters.

            Parameters:
            X_t (np.ndarray): Independent variables matrix X.
            Z_t (np.ndarray): Independent variables matrix Z.
            Y_t (np.ndarray): Dependent variable vector Y.
            train_ratio (float): Ratio of data to be used for training.
            val_ratio (float): Ratio of data to be used for validation.
            test_ratio (float): Ratio of data to be used for testing.
            """
            self.X_t = X_t
            self.Z_t = Z_t
            self.Y_t = Y_t
            self.epoch = config["modelTraining"]["epoch"]
            self.n_splits = config["modelTraining"]["nSplits"]
            self.train_ratio = train_ratio
            self.val_ratio = val_ratio
            self.test_ratio = test_ratio
            self.metrics = Metrics()
            self._split_data()

    def _split_data(self):
        """
        Split the data into training, validation, and test sets based on the provided ratios.
        """
        X_train, X_temp, Z_train, Z_temp, Y_train, Y_temp = train_test_split(
            self.X_t, self.Z_t, self.Y_t, test_size=(1 - self.train_ratio), random_state=42
        )
        val_test_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
        self.X_val, self.X_test, self.Z_val, self.Z_test, self.Y_val, self.Y_test = train_test_split(
            X_temp, Z_temp, Y_temp, test_size=val_test_ratio, random_state=42
        )
        self.X_train, self.Z_train, self.Y_train = X_train, Z_train, Y_train

    def train(self):
        """
        Train the model for a given number of epochs.

        Parameters:
        n_epochs (int): Number of epochs to train the model.
        """
        initial_params = self._initialize_parameters()

        if config['inferenceMethod'] == "torch_autograd":
            params = self._optimize(self.Y_train, self.X_train, self.Z_train, initial_params)
        
        elif config['inferenceMethod'] == "pertub_gradient_descent":
            params = GradientDescentPerturbation().optimize(self.Y_train, self.X_train, self.Z_train, initial_params)

        self.save_model(params)
        utils.log_parameters_results(params=params)
        # Evaluate on test set
        test_loss, test_mse = self._evaluate(params, self.X_test, self.Z_test, self.Y_test)
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_mse}")

        return params

    def train_cross_validation(self):
        """
        Train the model using cross-validation and evaluate on the test set.
        """
        initial_params = self._initialize_parameters()

        # kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        # val_losses = []
        # val_mse_list = []

        # train_losses = []
        # train_mse_list = []

        # for train_index, val_index in kf.split(self.X_train):
        #     X_train_cv, X_val_cv = self.X_train[train_index], self.X_train[val_index]
        #     Z_train_cv, Z_val_cv = self.Z_train[train_index], self.Z_train[val_index]
        #     Y_train_cv, Y_val_cv = self.Y_train[train_index], self.Y_train[val_index]

        #     if config['inferenceMethod'] == "torch_autograd":
        #         params = self._optimize(Y_train_cv, X_train_cv, Z_train_cv, initial_params)
        #     elif config['inferenceMethod'] == "pertub_gradient_descent":
        #         params = GradientDescentPerturbation().optimize(Y_train_cv, X_train_cv, Z_train_cv, initial_params)

        #     train_loss, train_mse = self._evaluate(params, X_train_cv, Z_train_cv, Y_train_cv)
        #     val_loss, val_mse = self._evaluate(params, X_val_cv, Z_val_cv, Y_val_cv)
            
        #     train_losses.append(train_loss)
        #     train_mse_list.append(train_mse)
        #     val_losses.append(val_loss)
        #     val_mse_list.append(val_mse)

        # avg_val_loss = np.mean(val_losses)
        # avg_val_mse = np.mean(val_mse_list)
        # print(f"Average validation loss from cross-validation: {avg_val_loss}")
        # print(f"Average validation accuracy from cross-validation: {avg_val_mse}")

        # Train on full training data
        if config['inferenceMethod'] == "torch_autograd":
            params = self._optimize(self.Y_train, self.X_train, self.Z_train, initial_params)
        elif config['inferenceMethod'] == "pertub_gradient_descent":
            params = GradientDescentPerturbation().optimize(self.Y_train, self.X_train, self.Z_train, initial_params)

        self.save_model(params)
        utils.log_parameters_results(params=params)

        # Evaluate on test set
        test_loss, test_mse = self._evaluate(params, self.X_test, self.Z_test, self.Y_test)
        print(f"Test loss: {test_loss}")
        print(f"Test accuracy: {test_mse}")

        return params
    
    def _optimize(self, Y_t, X_t, Z_t, params):
        """
        Optimize the parameters using PyTorch's autograd and SGD optimizer.

        Parameters:
        Y_t (torch.Tensor): Dependent variable vector Y.
        X_t (torch.Tensor): Independent variables matrix X.
        Z_t (torch.Tensor): Independent variables matrix Z.
        params (list): List of initial parameters (G, eta, zeta).

        Returns:
        list: Optimized parameters.
        """
        optimizer = optim.SGD(params, lr=config["modelTraining"]["learningRate"])
        
        train_losses = []
        val_losses = []
        train_mse_list = []
        val_mse_list = []

        for epoch in range(self.epoch):
            optimizer.zero_grad()
            loss = utils.negative_log_likelihood(params, Y_t, X_t, Z_t)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warn("NaN or inf encountered in loss. Exiting.")
                break
            
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                logging.info(f"epoch {epoch}: Loss = {loss.item()}")
                
            # Track metrics (dummy implementation, replace with actual evaluation)
            train_loss, train_mse = self._evaluate(params, X_t, Z_t, Y_t)
            val_loss, val_mse = self._evaluate(params, self.X_val, self.Z_val, self.Y_val)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_mse_list.append(train_mse)
            val_mse_list.append(val_mse)

        self._plot_metrics(train_losses, val_losses, train_mse_list, val_mse_list)

        return params

    def _initialize_parameters(self):
        """
        Initialize parameters and add noise based on the inference method specified in the config.

        Returns:
        list: Initial parameters with added noise.
        """
        initial_G = np.random.uniform(0, 1)
        initial_eta = np.random.normal(size=self.X_t.shape[1])
        initial_zeta = np.random.normal(size=self.Z_t.shape[1])

        logging.info(f"initial_G: {initial_G}")
        logging.info(f"initial_eta: {initial_eta}")
        logging.info(f"initial_zeta: {initial_zeta}")

        if config['inferenceMethod'] == 'torch_autograd':
            # Convert initial parameters to PyTorch tensors with gradients enabled
            initial_G_tensor = torch.tensor(initial_G, dtype=torch.float32, requires_grad=True)
            initial_eta_tensor = torch.tensor(initial_eta, dtype=torch.float32, requires_grad=True)
            initial_zeta_tensor = torch.tensor(initial_zeta, dtype=torch.float32, requires_grad=True)
            initial_params = [nn.Parameter(initial_G_tensor), nn.Parameter(initial_eta_tensor), nn.Parameter(initial_zeta_tensor)]
        elif config['inferenceMethod'] == 'pertub_gradient_descent':
            # Combine initial parameters into a single list
            initial_params = [initial_G] + list(initial_eta) + list(initial_zeta)
        
        return initial_params

    def save_model(self, params):
        """
        Save the model parameters to a file.

        Parameters:
        params : model parameters to be saved.
        """
        with open(config["modelTraining"]["modelPath"], 'wb') as file:
            if config['inferenceMethod'] == 'torch_autograd':
                G = params[0].detach().numpy()
                eta = params[1].detach().numpy()
                zeta = params[2].detach().numpy()

            elif config['inferenceMethod'] == 'pertub_gradient_descent':
                G, *coeffs = params
                eta = np.array(coeffs[:self.X_t.shape[1]])
                zeta = np.array(coeffs[self.X_t.shape[1]:])

            pickle.dump({
                'G': G,
                'eta': eta,
                'zeta': zeta,
            }, file)


    def _evaluate(self, params, X, Z, Y):
        """
        Evaluate the model on the given data.

        Parameters:
        params : model parameters.
        X (np.ndarray): Independent variables matrix X.
        Z (np.ndarray): Independent variables matrix Z.
        Y (np.ndarray): Dependent variable vector Y.

        Returns:
        float: The average loss.
        """
        if config['inferenceMethod'] == 'torch_autograd':
            G = params[0].detach().numpy()
            eta = params[1].detach().numpy()
            zeta = params[2].detach().numpy()
        elif config['inferenceMethod'] == 'pertub_gradient_descent':
            G, *coeffs = params
            eta = np.array(coeffs[:self.X_t.shape[1]])
            zeta = np.array(coeffs[self.X_t.shape[1]:])

        model = DynamicLinearModel()
        model.G = G
        model.eta = eta
        model.zeta = zeta
        model.T = X.shape[0]

        predicted_Y = model.dlm_model(X, Z)
        loss = np.mean((predicted_Y - Y) ** 2)
        mse = self.metrics.mse(Y, predicted_Y)
        return loss, mse


    def _plot_metrics(self, train_losses, val_losses, train_mse_list, val_mse_list):
        """
        Plot training and validation loss and accuracy, and test metrics.

        Parameters:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        test_loss (float): Test loss.
        train_mse_list (list): List of training accuracies.
        val_mse_list (list): List of validation accuracies.
        test_mse (float): Test accuracy.
        """
        plt.figure(figsize=(14, 6))

        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epoch), train_losses, 'b-', label='Training loss')
        plt.plot(range(self.epoch), val_losses, 'r-', label='Validation loss')
        plt.title('Training, Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(range(self.epoch), train_mse_list, 'b-', label='Training mse')
        plt.plot(range(self.epoch), val_mse_list, 'r-', label='Validation mse')
        plt.title('Training, Validation MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()

        plt.tight_layout()
