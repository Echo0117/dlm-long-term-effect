import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
from config import config
import dynamic_linear_model.utils as utils
import logging
import pickle

from matplotlib import pyplot as plt
from config import config
import torch.nn as nn
import torch.optim as optim
import dynamic_linear_model.utils as utils
from dynamic_linear_model.model import DynamicLinearModel
from dynamic_linear_model.evaluations.metrics import Metrics


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
        self.obtain_parameters_by_lr()
        self.save_simulated_parameters()

    def obtain_parameters_by_lr(self):
        """
        Obtain eta (η) and zeta (ζ) by fitting a linear regression model to XZ_t and Y_t.
        """
        XZ_t = np.hstack((self.X_t, self.Z_t))
        model_XZ = LinearRegression().fit(XZ_t, self.Y_t)
        eta_zeta = model_XZ.coef_
        
        self.eta = eta_zeta[:self.X_t.shape[1]]
        self.zeta = eta_zeta[self.X_t.shape[1]:]
        print("self.eta", self.eta)
        print("self.zeta", self.zeta)

    def save_simulated_parameters(self):
        """
        Save the simulated parameters to a CSV file.
        """
        simulated_params = {
            "Param": ["G"] + [f"η{i+1}" for i in range(len(self.eta))] + [f"ζ{i+1}" for i in range(len(self.zeta))],
            "Simulated": [self.G] + list(self.eta) + list(self.zeta)
        }
        df_simulated = pd.DataFrame(simulated_params)
        df_simulated.to_csv(config["simulationRecovery"]["paramsSavedPath"], index=False)

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



class SimulationRecovery():
    def __init__(self, X_t, Z_t, Y_t):
            """
            Initialize the Trainer class with the given parameters.

            Parameters:
            X_t (np.ndarray): Independent variables matrix X.
            Z_t (np.ndarray): Independent variables matrix Z.
            Y_t (np.ndarray): Dependent variable vector Y.
            """
            self.X_t = X_t
            self.Z_t = Z_t
            self.Y_t = Y_t
            self.epoch = config["modelTraining"]["epoch"]
            self.n_splits = config["modelTraining"]["nSplits"]
            self.metrics = Metrics()
    
    def recovery_for_simulation(self):
        """
        Train the model for multiple iterations and record statistics of parameters.
        """
        params_list = []
        losses = []
        Y_predicted_list = []
        for _ in range(config["simulationRecovery"]["iteration"]):
            print("iteration: ", _)
            initial_params = self._initialize_parameters()

            params = self._optimize(self.Y_t, self.X_t, self.Z_t, initial_params)

            loss, mse, Y_predicted = self._evaluate(params, self.X_t, self.Z_t, self.Y_t)
            params_list.append(params)
            losses.append(loss)
            Y_predicted_list.append(Y_predicted)

        # Calculate statistics
        self._calculate_statistics(params_list, losses)
        return Y_predicted_list
    
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
        
        losses = []
        mse_list = []

        for epoch in range(self.epoch):
            optimizer.zero_grad()
            loss = utils.negative_log_likelihood(params, Y_t, X_t, Z_t)
            
            # if torch.isnan(loss) or torch.isinf(loss):
            #     logging.warn("NaN or inf encountered in loss. Exiting.")
            #     break
            
            loss.backward()
            # Clip gradients to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            if epoch % 100 == 0:
                logging.info(f"epoch {epoch}: Loss = {loss.item()}")
                
            loss, mse, _ = self._evaluate(params, X_t, Z_t, Y_t)
            
            losses.append(loss)
            mse_list.append(mse)

        # self._plot_metrics(losses, mse_list)

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
        
        try:
            mse = self.metrics.mse(Y, predicted_Y)
        except Exception as e:
            print(e)
            print("Y: ", Y)
            print("predicted_Y: ", predicted_Y)
            print("loss: ", loss)
            print("G: ", G)
            print("eta: ", eta)
            print("zeta: ", zeta)
        return loss, mse, predicted_Y


    def _plot_metrics(self, losses, mse_list):
        """
        Plot training and validation loss and accuracy, and test metrics.

        Parameters:
        losses (list): List of losses.
        mse_list (list): List of mse.
        """
        plt.figure(figsize=(14, 6))
        print("range(self.epoch)", range(self.epoch))
        print("losses", losses)
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epoch), losses, 'b-', label='Training loss')
        plt.title('Loss during Recovery')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot mse
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epoch), mse_list, 'r-', label='Training mse')
        plt.title('MSE during Recovery')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()

        plt.tight_layout()

    def _calculate_statistics(self, params_list, losses):
        """
        Calculate and print the statistics of the parameters from multiple runs.
        """
        Gs, etas, zetas = [], [], []

        for params in params_list:
            if config['inferenceMethod'] == 'torch_autograd':
                G = params[0].detach().numpy()
                eta = params[1].detach().numpy()
                zeta = params[2].detach().numpy()
            elif config['inferenceMethod'] == 'pertub_gradient_descent':
                G, *coeffs = params
                eta = np.array(coeffs[:self.X_t.shape[1]])
                zeta = np.array(coeffs[self.X_t.shape[1]:])
            
            Gs.append(G)
            etas.append(eta)
            zetas.append(zeta)

        # Calculate statistics
        Gs = np.array(Gs)
        etas = np.array(etas)
        zetas = np.array(zetas)

        G_stats = {
            'initial': Gs[0],
            'mean': np.mean(Gs),
            'std': np.std(Gs),
            'median': np.median(Gs),
            'max': np.max(Gs)
        }

        eta_stats = {
            'initial': etas[0],
            'mean': np.mean(etas, axis=0),
            'std': np.std(etas, axis=0),
            'median': np.median(etas, axis=0),
            'max': np.max(etas, axis=0)
        }

        zeta_stats = {
            'initial': zetas[0],
            'mean': np.mean(zetas, axis=0),
            'std': np.std(zetas, axis=0),
            'median': np.median(zetas, axis=0),
            'max': np.max(zetas, axis=0)
        }

        print("G statistics:", G_stats)
        print("Eta statistics:", eta_stats)
        print("Zeta statistics:", zeta_stats)


        # Combine all statistics and iteration results into a single DataFrame
        rows = [["G", G_stats['initial'], *Gs.tolist(), G_stats['mean'], G_stats['std'], G_stats['median'], G_stats['max']]]

        for i in range(len(eta_stats['initial'])):
            rows.append([f"η{i+1}", eta_stats['initial'][i], *etas[:, i].tolist(), eta_stats['mean'][i], eta_stats['std'][i], eta_stats['median'][i], eta_stats['max'][i]])

        for i in range(len(zeta_stats['initial'])):
            rows.append([f"ζ{i+1}", zeta_stats['initial'][i], *zetas[:, i].tolist(), zeta_stats['mean'][i], zeta_stats['std'][i], zeta_stats['median'][i], zeta_stats['max'][i]])

        columns = ["Param", "Initial"] + [f"iteration {i+1}" for i in range(len(params_list))] + ["Mean", "Std", "Median", "Max"]
        df = pd.DataFrame(rows, columns=columns)

        df_simulated = pd.read_csv(config["simulationRecovery"]["paramsSavedPath"])
        df["Simulated"] = df_simulated["Simulated"]
    
        # Save to CSV
        df.to_csv(config["simulationRecovery"]["paramsSavedPath"], index=False)

        print("Simulation results saved to simulation_results.csv")