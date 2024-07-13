import logging
import os
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
from dynamic_linear_model.inference.samplying_method import MCMCSamplingMethod
import multiprocessing
from multiprocessing import Pool

class ClippedAdam(torch.optim.Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        super(ClippedAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def step(self, closure=None):
        super(ClippedAdam, self).step(closure)
        with torch.no_grad():
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        param.data.clamp_(0, 1)


class SimulationRecovery:
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

    # def run_simulation_task(self, run_id):
    #     """
    #     Function to run a single simulation task. This will be executed in parallel.
    #     """
    #     print("independentRun: ", run_id)
    #     initial_params = self._initialize_parameters()
    #     try:
    #         params = self._optimize(self.Y_t, self.X_t, self.Z_t, initial_params)
    #     except UnboundLocalError as e:
    #         logging.error(e)
    #         return None, None, None

    #     Y_predicted = self._get_predicted_Y(params, self.X_t, self.Z_t, self.Y_t)
    #     return params, Y_predicted, params[0].detach().numpy()

    # def recovery_for_simulation(self):
    #     """
    #     Train the model for multiple iterations and record statistics of parameters.
    #     """
    #     num_runs = config["simulationRecovery"]["independentRun"]

    #     with Pool(processes=multiprocessing.cpu_count()) as pool:
    #         results = pool.starmap(self.run_simulation_task, [(i,) for i in range(num_runs)])

    #     # Filter out failed runs
    #     results = [res for res in results if res[0] is not None]

    #     params_list = [res[0] for res in results]
    #     Y_predicted_list = [res[1] for res in results]
    #     g_list = [res[2] for res in results]

    #     # Calculate statistics
    #     self._calculate_statistics(params_list)
    #     self._plot_params(params_list)
    #     return Y_predicted_list

    def recovery_for_simulation(self, ax, ax_training, ax_optim_g):
        """
        Train the model for multiple iterations and record statistics of parameters.
        """
        g_list = []
        params_list = []

        Y_predicted_list = []

        for num_run, ax_training_sub, ax_optim_g_sub in zip(
            range(config["simulationRecovery"]["independentRun"]), ax_training, ax_optim_g
        ):
            print("independentRun: ", num_run)
            initial_params = self._initialize_parameters()
            try:
                params = self._optimize(
                    self.Y_t,
                    self.X_t,
                    self.Z_t,
                    initial_params,
                    ax_training_sub,
                    ax_optim_g_sub,
                    num_run,
                )
            except UnboundLocalError as e:
                logging.error(e)
                self._plot_params(params_list)
                break

            Y_predicted = self._get_predicted_Y(params, self.X_t, self.Z_t, self.Y_t)

            params_list.append(params)
            Y_predicted_list.append(Y_predicted)
            g_list.append(params[0].detach().numpy())
            self.save_model(params)

        # Calculate statistics
        self._calculate_statistics(params_list)
        self._plot_params(params_list, ax)
        return Y_predicted_list

    def _optimize(self, Y_t, X_t, Z_t, params, ax, ax_1, num_run):
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

        if config["inferenceMethod"] == "torch_autograd":
            print("before optim", params)
            # optimizer = optim.SGD(
            #     params,
            #     lr=config["modelTraining"]["learningRate"],
            #     # momentum=0.1,  # Momentum
            #     # weight_decay=1e-5,  # Weight decay for regularization
            #     # nesterov=True,  # Use Nesterov Accelerated Gradient
            # )
            optimizer = optim.Adam(params, lr=config["modelTraining"]["learningRate"], weight_decay=0.001)
            losses = []
            params_before = []
            params_after_optim = []
            params_after_sigmoid = []
            for epoch in range(self.epoch):

                loss = utils.negative_log_likelihood(params, Y_t, X_t, Z_t)
                params_before.append(params[0].data.clone().numpy())
                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                params_after_optim.append(params[0].data.clone().numpy())
                if config["modelTraining"]["addSigmoid"]:
                    params[0].data = torch.sigmoid(params[0].data)

                params_after_sigmoid.append(params[0].data.clone().numpy())

                if epoch % 100 == 0:
                    logging.info(f"epoch {epoch}: Loss = {loss.item()}")

                losses.append(loss.item())
                # mse_list.append(mse)

            self._plot_metrics(losses, ax, num_run)
            self._plot_g(params_before, params_after_optim, params_after_sigmoid, ax_1)
            # After training, ensure that the parameters have been updated
            for i, param in enumerate(params):
                logging.info(f"Paramssss {i}: {param.data}")

        elif config["inferenceMethod"] == "mcmc":
            params = MCMCSamplingMethod().extract_parameters(params, Y_t, X_t, Z_t)

        print("after optim", params)
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

        if config["inferenceMethod"] == "torch_autograd":
            # Convert initial parameters to PyTorch tensors with gradients enabled
            initial_G_tensor = torch.tensor(
                initial_G, dtype=torch.float32, requires_grad=True
            )
            initial_eta_tensor = torch.tensor(
                initial_eta, dtype=torch.float32, requires_grad=True
            )
            initial_zeta_tensor = torch.tensor(
                initial_zeta, dtype=torch.float32, requires_grad=True
            )
            initial_params = [
                nn.Parameter(initial_G_tensor),
                nn.Parameter(initial_eta_tensor),
                nn.Parameter(initial_zeta_tensor),
            ]
        elif config["inferenceMethod"] == "pertub_gradient_descent":
            # Combine initial parameters into a single list
            initial_params = [initial_G] + list(initial_eta) + list(initial_zeta)
        elif config["inferenceMethod"] == "mcmc":
            initial_params = np.append(
                np.append(np.array([initial_G]), initial_eta), initial_zeta
            )
        else:
            raise ValueError("Invalid inference method specified in the config.")
        return initial_params

    def save_model(self, params):
        """
        Save the model parameters to a file.

        Parameters:
        params : model parameters to be saved.
        """
        with open(config["modelTraining"]["modelPath"], "wb") as file:
            if config["inferenceMethod"] == "torch_autograd":
                G = params[0].detach().numpy()
                eta = params[1].detach().numpy()
                zeta = params[2].detach().numpy()

            elif config["inferenceMethod"] == "pertub_gradient_descent":
                G, *coeffs = params
                eta = np.array(coeffs[: self.X_t.shape[1]])
                zeta = np.array(coeffs[self.X_t.shape[1] :])

            pickle.dump(
                {
                    "G": G,
                    "eta": eta,
                    "zeta": zeta,
                },
                file,
            )

    def _get_predicted_Y(self, params, X, Z, Y):
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
        if (
            config["inferenceMethod"] == "torch_autograd"
            or config["inferenceMethod"] == "mcmc"
        ):
            G = params[0].detach().numpy()
            eta = params[1].detach().numpy()
            zeta = params[2].detach().numpy()
        elif config["inferenceMethod"] == "pertub_gradient_descent":
            G, *coeffs = params
            eta = np.array(coeffs[: self.X_t.shape[1]])
            zeta = np.array(coeffs[self.X_t.shape[1] :])

        model = DynamicLinearModel()
        model.G = G
        model.eta = eta
        model.zeta = zeta
        model.T = X.shape[0]

        predicted_Y = model.dlm_model(X, Z)

        return predicted_Y

    def _plot_metrics(self, losses, ax, num_run):
        """
        Plot training and validation loss and accuracy, and test metrics.

        Parameters:
        losses (list): List of losses.
        mse_list (list): List of mse.
        """
        ax.plot(range(self.epoch), losses, "b-", label="Training loss")
        ax.set_title(
            f'Independent run {num_run} with G = {"{:.1f}".format(config["modelTraining"]["originalG"])}',
        )

        # # Plot mse
        # plt.subplot(1, 2, 1)
        # plt.plot(range(self.epoch), mse_list, 'r-', label='Training mse')
        # plt.title('MSE during Recovery')
        # plt.xlabel('Epoch')
        # plt.ylabel('MSE')
        # plt.legend()

        # plt.tight_layout()

    def _plot_g(self, params_before, params_after_optim, params_after_sigmoid, ax):
        """
        Plot the G values for each run.
        """
        ax.plot(range(config["modelTraining"]["epoch"]), params_before, label='Before optim')
        ax.plot(range(config["modelTraining"]["epoch"]), params_after_optim, label='After optim')
        if config["modelTraining"]["addSigmoid"]:
            ax.plot(range(config["modelTraining"]["epoch"]), params_after_sigmoid, label='After optim and Sigmoid', linestyle='--')

        # ax.legend()
        # plt.show()

    def _plot_params(self, params_list, ax):
        """
        Plot the G, eta, and zeta values for each run.
        """
        # num_runs = len(params_list)
        num_epochs = len(params_list)

        # plt.figure(figsize=(14, 6))
        Gs, etas, zetas = [], [], []
        for i, params in enumerate(params_list):
            Gs.append(params[0].detach().numpy())
            etas.append(params[1].detach().numpy())
            zetas.append(params[2].detach().numpy())

        # Plot G values
        ax.plot(range(num_epochs), Gs, label=f"G")
        etas = np.array(etas).T
        zetas = np.array(zetas).T
        # Plot eta values
        for j in range(etas.shape[0])[:2]:
            ax.plot(range(num_epochs), etas[j], label=f"eta {j+1}")

        # Plot zeta values
        for k in range(zetas.shape[0])[:2]:
            ax.plot(range(num_epochs), zetas[k], label=f"zeta {k+1}")

    def _calculate_statistics(self, params_list):
        """
        Calculate and print the statistics of the parameters from multiple runs.
        """
        Gs, etas, zetas = [], [], []

        for params in params_list:
            if (
                config["inferenceMethod"] == "torch_autograd"
                or config["inferenceMethod"] == "mcmc"
            ):
                G = params[0].detach().numpy()
                eta = params[1].detach().numpy()
                zeta = params[2].detach().numpy()
            elif config["inferenceMethod"] == "pertub_gradient_descent":
                G, *coeffs = params
                eta = np.array(coeffs[: self.X_t.shape[1]])
                zeta = np.array(coeffs[self.X_t.shape[1] :])

            Gs.append(G)
            etas.append(eta)
            zetas.append(zeta)

        # Calculate statistics
        Gs = np.array(Gs)
        etas = np.array(etas)
        zetas = np.array(zetas)

        G_stats = {
            "mean": np.mean(Gs),
            "std": np.std(Gs),
            "median": np.median(Gs),
            "max": np.max(Gs),
        }

        eta_stats = {
            "mean": np.mean(etas, axis=0),
            "std": np.std(etas, axis=0),
            "median": np.median(etas, axis=0),
            "max": np.max(etas, axis=0),
        }

        zeta_stats = {
            "mean": np.mean(zetas, axis=0),
            "std": np.std(zetas, axis=0),
            "median": np.median(zetas, axis=0),
            "max": np.max(zetas, axis=0),
        }

        print("G statistics:", G_stats)
        print("Eta statistics:", eta_stats)
        print("Zeta statistics:", zeta_stats)

        # Combine all statistics and iteration results into a single DataFrame
        rows = [
            [
                "G",
                *Gs.tolist(),
                G_stats["mean"],
                G_stats["std"],
                G_stats["median"],
                G_stats["max"],
            ]
        ]

        for i in range(len(eta_stats["mean"])):
            rows.append(
                [
                    f"η{i+1}",
                    *etas[:, i].tolist(),
                    eta_stats["mean"][i],
                    eta_stats["std"][i],
                    eta_stats["median"][i],
                    eta_stats["max"][i],
                ]
            )

        for i in range(len(zeta_stats["mean"])):
            rows.append(
                [
                    f"ζ{i+1}",
                    *zetas[:, i].tolist(),
                    zeta_stats["mean"][i],
                    zeta_stats["std"][i],
                    zeta_stats["median"][i],
                    zeta_stats["max"][i],
                ]
            )

        columns = (
            ["Param"]
            + [f"iteration {i+1}" for i in range(len(params_list))]
            + ["Mean", "Std", "Median", "Max"]
        )
        df = pd.DataFrame(rows, columns=columns)

        # Define the file path from the configuration
        file_path = config["simulationRecovery"]["paramsSavedPath"]
        file_simulated_path = config["simulationRecovery"]["simulatedParamSavedPath"]

        df_existing = (
            pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()
        )
        df_simulated_existing = (
            pd.read_csv(file_simulated_path)
            if os.path.exists(file_simulated_path)
            else pd.DataFrame()
        )

        df["Simulated"] = df_simulated_existing["Simulated"][: len(rows)]
        new_df = pd.concat([df, df_existing], axis=0)
        empty_row = pd.DataFrame([[None] * len(new_df.columns)], columns=new_df.columns)
        new_df = pd.concat([new_df, empty_row], ignore_index=True)

        # Save to CSV
        new_df.to_csv(config["simulationRecovery"]["paramsSavedPath"], index=False)

        print("Simulation results saved to simulation_results.csv")

    # def _calculate_statistics(self, params_list):
    #     """
    #     Calculate and print the statistics of the parameters from multiple runs.
    #     """
    #     Gs, etas, zetas = [], [], []

    #     for params in params_list:
    #         if config['inferenceMethod'] == 'torch_autograd' or config['inferenceMethod'] == 'mcmc':
    #             G = params[0].detach().numpy()
    #             eta = params[1].detach().numpy()
    #             zeta = params[2].detach().numpy()
    #         elif config['inferenceMethod'] == 'pertub_gradient_descent':
    #             G, *coeffs = params
    #             eta = np.array(coeffs[:self.X_t.shape[1]])
    #             zeta = np.array(coeffs[self.X_t.shape[1]:])

    #         Gs.append(G)
    #         etas.append(eta)
    #         zetas.append(zeta)

    #     # Calculate statistics
    #     Gs = np.array(Gs)
    #     etas = np.array(etas)
    #     zetas = np.array(zetas)

    #     G_stats = {
    #         'mean': np.mean(Gs),
    #         'std': np.std(Gs),
    #         'median': np.median(Gs),
    #         'max': np.max(Gs)
    #     }

    #     eta_stats = {
    #         'mean': np.mean(etas, axis=0),
    #         'std': np.std(etas, axis=0),
    #         'median': np.median(etas, axis=0),
    #         'max': np.max(etas, axis=0)
    #     }

    #     zeta_stats = {
    #         'mean': np.mean(zetas, axis=0),
    #         'std': np.std(zetas, axis=0),
    #         'median': np.median(zetas, axis=0),
    #         'max': np.max(zetas, axis=0)
    #     }

    #     print("G statistics:", G_stats)
    #     print("Eta statistics:", eta_stats)
    #     print("Zeta statistics:", zeta_stats)

    #     # Combine all statistics and iteration results into a single DataFrame
    #     rows = [["G", *Gs.tolist(), G_stats['mean'], G_stats['std'], G_stats['median'], G_stats['max']]]

    #     for i in range(len(eta_stats['mean'])):
    #         rows.append([f"η{i+1}", *etas[:, i].tolist(), eta_stats['mean'][i], eta_stats['std'][i], eta_stats['median'][i], eta_stats['max'][i]])

    #     for i in range(len(zeta_stats['mean'])):
    #         rows.append([f"ζ{i+1}", *zetas[:, i].tolist(), zeta_stats['mean'][i], zeta_stats['std'][i], zeta_stats['median'][i], zeta_stats['max'][i]])

    #     columns = ["Param"] + [f"iteration {i+1}" for i in range(len(params_list))] + ["Mean", "Std", "Median", "Max"]
    #     df = pd.DataFrame(rows, columns=columns)

    #     df_simulated = pd.read_csv(config["simulationRecovery"]["paramsSavedPath"])
    #     df["Simulated"] = df_simulated["Simulated"]

    #     # Save to CSV
    #     df.to_csv(config["simulationRecovery"]["paramsSavedPath"], index=False)

    #     print("Simulation results saved to simulation_results.csv")


class DataSimulation:
    def __init__(
        self,
        X_t: np.ndarray,
        Z_t: np.ndarray,
        Y_t: np.ndarray,
        G: float = config["modelTraining"]["originalG"],
        theta_0: float = config["modelTraining"]["theta0"],
    ):
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

        self.eta = eta_zeta[: self.X_t.shape[1]] 
        self.zeta = eta_zeta[self.X_t.shape[1] :] 
        print("self.eta", self.eta)
        print("self.zeta", self.zeta)

    def save_simulated_parameters(self):
        """
        Save the simulated parameters to a CSV file by adding new rows above the original file.
        """
        simulated_params = {
            "Param": ["G"]
            + [f"η{i+1}" for i in range(len(self.eta))]
            + [f"ζ{i+1}" for i in range(len(self.zeta))],
            "Simulated": [self.G] + list(self.eta) + list(self.zeta),
        }
        df_simulated = pd.DataFrame(simulated_params)

        file_path = config["simulationRecovery"]["simulatedParamSavedPath"]

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([df_simulated, existing_df], ignore_index=True)
        else:
            combined_df = df_simulated

        combined_df.to_csv(file_path, index=False)

    # def generate_theta(self) -> np.ndarray:
    #     """
    #     Generate theta values using the state transition equation.

    #     :return: Array of theta values.
    #     """
    #     T = len(self.Y_t)
    #     theta_t = np.zeros(T)
    #     theta_t[0] = self.theta_0
    #     print("self.G_simulatioln", self.G)
    #     for t in range(T):
    #         if t > 0:
    #             theta_t[t] = self.G * theta_t[t-1] + np.dot(self.Z_t[t-1], self.zeta / 2)

    #     return theta_t

    # def generate_simulated_Y(self) -> np.ndarray:
    #     """
    #     Generate simulated Y values using the DLM model.

    #     :return: Array of predicted Y values.
    #     """
    #     self.obtain_parameters_by_lr()
    #     T = len(self.Y_t)
    #     theta_t = self.generate_theta()
    #     predicted_Y = np.zeros(T)

    #     for t in range(T):
    #         predicted_Y[t] = theta_t[t] + np.dot(self.X_t[t], self.eta) + np.dot(self.Z_t[t], self.zeta / 2)
    #     return predicted_Y
    def generate_simulated_Y(self):
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
        T = len(self.X_t)
        theta_t = np.zeros(T)
        predicted_Y = np.zeros(T)
        new = np.zeros(T)
        for t in range(T):
            if t > 0:
                # State transition equation
                theta_t[t] = self.G * theta_t[t - 1] + np.dot(
                    self.Z_t[t - 1], self.zeta / 2
                )
            # Observation equation
            predicted_Y[t] = (
                theta_t[t]
                + np.dot(self.X_t[t], self.eta)
                + np.dot(self.Z_t[t], self.zeta / 2)
            )
            new[t] = np.dot(self.X_t[t], self.eta) + np.dot(self.Z_t[t], self.zeta / 2)
            print("predicted_Y[t]predicted_Y[t]", predicted_Y[t])
        return predicted_Y

    def get_simulation_results(self) -> dict:
        """
        Get a DataFrame with actual and predicted Y values for visualization.

        :return: DataFrame containing actual and predicted Y values.
        """
        predicted_Y = self.generate_simulated_Y()
        results_Y = {"actual_Y": self.Y_t, "simulated_Y": predicted_Y}
        utils.log_parameters_results(self.G, self.eta, self.zeta)
        return results_Y
