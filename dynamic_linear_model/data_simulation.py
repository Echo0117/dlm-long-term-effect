import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import config
from dynamic_linear_model.model import DynamicLinearModel
import dynamic_linear_model.utils as utils


device = config["device"]


class SimulationRecovery:
    def __init__(self, X_t: np.ndarray, Z_t: np.ndarray, Y_t: np.ndarray):
        """
        Initialize the SimulationRecovery class with the given parameters.

        Parameters:
        X_t (np.ndarray): Independent variables matrix X.
        Z_t (np.ndarray): Independent variables matrix Z.
        Y_t (np.ndarray): Dependent variable vector Y.
        """
        self.X_t = torch.tensor(X_t, dtype=torch.float32, device=device)
        self.Z_t = torch.tensor(Z_t, dtype=torch.float32, device=device)
        self.Y_t = torch.tensor(Y_t, dtype=torch.float32, device=device)
        self.epoch = config["modelTraining"]["epoch"]
        self.n_splits = config["modelTraining"]["nSplits"]
        self.model = DynamicLinearModel().to(device)

    def run_optimization(self, num_run: int) -> tuple:
        """
        Run the optimization for a single independent run.

        Parameters:
        num_run (int): The current run number.

        Returns:
        tuple: A tuple containing model parameters, predicted Y, optimization metrics, and final loss.
        """

        print("independentRun: ", num_run)
        self.model = DynamicLinearModel().to(device)
        try:
            params_before, params_after_optim, losses = self._optimize(
                self.Y_t,
                self.X_t,
                self.Z_t,
            )
        except UnboundLocalError as e:
            logging.error(e)
            return None, None

        Y_predicted = self._get_predicted_Y(self.X_t, self.Z_t)
        final_loss = losses[-1]
        return (
            list(self.model.named_parameters()),
            Y_predicted.data.cpu().numpy(),
            (params_before, params_after_optim, losses),
            final_loss,
        )

    def recovery_for_simulation(
        self, ax: Axes, ax_training: Axes, ax_optim_g: Axes
    ) -> np.ndarray:
        """
        Train the model for multiple iterations and record statistics of parameters.

        Parameters:
        ax (matplotlib.axes.Axes): Axes for plotting parameter statistics.
        ax_training (matplotlib.axes.Axes): Axes for plotting training metrics.
        ax_optim_g (matplotlib.axes.Axes): Axes for plotting optimization metrics.

        Returns:
        np.ndarray: The best predicted Y values.
        """
        num_runs = config["simulationRecovery"]["independentRun"]
        params_list, metrics_list = [], []
        best_Y_predicted = None
        best_final_loss = float("inf")
        plotter = utils.Plotter()

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.run_optimization, num_run)
                for num_run in range(num_runs)
            ]

            for future in as_completed(futures):
                params, Y_predicted, metrics, final_loss = future.result()
                if params is not None and Y_predicted is not None:
                    params_list.append(params)
                    metrics_list.append(metrics)
                    if final_loss < best_final_loss:
                        best_final_loss = final_loss
                        best_Y_predicted = Y_predicted
                        best_run_number = futures.index(future)

        # Plot metrics after multiprocessing
        plotter.plot_metrics_multiprocess(metrics_list, ax_training, ax_optim_g)

        # Calculate statistics
        Gs, etas, zetas, gammas = plotter.plot_params(params_list, ax)
        utils.calculate_statistics(Gs, etas, zetas, gammas, best_run_number)

        return best_Y_predicted
 
    def mse_loss(self, predicted_Y: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Define the Mean Squared Error (MSE) loss function.

        Parameters:
        predicted_Y (torch.Tensor): Predicted Y values.
        Y (torch.Tensor): Actual Y values.

        Returns:
        torch.Tensor: The MSE loss.
        """
        return 0.5 * torch.mean((predicted_Y - Y) ** 2)

    def _optimize(
        self, Y_t: torch.Tensor, X_t: torch.Tensor, Z_t: torch.Tensor
    ) -> tuple:
        """
        Train the Dynamic Linear Model.

        Parameters:
        Y_t (torch.Tensor): Dependent variable vector Y.
        X_t (torch.Tensor): Independent variables matrix X.
        Z_t (torch.Tensor): Independent variables matrix Z.

        Returns:
        tuple: A tuple containing parameters before and after optimization and the list of losses.
        """
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=config["modelTraining"]["learningRate"],
            momentum=config["modelTraining"]["momentum"],
            weight_decay=config["modelTraining"]["weightDecay"]
        )

        losses, params_before, params_after_optim = [], [], []
        self.model.train()
        for epoch in range(self.epoch):
            params_before.append(self.model.G.item())

            optimizer.zero_grad()
            outputs, G, _, _, _ = self.model.forward(X_t, Z_t)
            loss = self.mse_loss(outputs, Y_t)
            loss.backward()
            optimizer.step()

            params_after_optim.append(G)
            losses.append(loss.item())

            if epoch % 100 == 0:
                logging.info(f"epoch {epoch}: Loss = {loss.item()}")

        for i, param in enumerate(self.model.parameters()):
            logging.info(f"Paramssss {i}: {param.data}")

        return params_before, params_after_optim, losses

    def _get_predicted_Y(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the model on the given data.

        Parameters:
        X (torch.Tensor): Independent variables matrix X.
        Z (torch.Tensor): Independent variables matrix Z.

        Returns:
        torch.Tensor: Predicted Y values.
        """
        predicted_Y, _, _, _, _ = self.model(X, Z)
        return predicted_Y
    

class DataSimulation:
    def __init__(
        self,
        X_t: np.ndarray,
        Z_t: np.ndarray,
        Y_t: np.ndarray,
        G: float = 0,
    ):
        """
        Initialize the DataSimulation class with data and parameters.

        Parameters:
        X_t (np.ndarray): Independent variables matrix X.
        Z_t (np.ndarray): Independent variables matrix Z.
        Y_t (np.ndarray): Dependent variable vector Y.
        G (float, optional): State transition coefficient. Default is 0.
        """
        self.X_t = torch.tensor(X_t, dtype=torch.float32, device=device)
        self.Z_t = torch.tensor(Z_t, dtype=torch.float32, device=device)
        self.Y_t = torch.tensor(Y_t, dtype=torch.float32, device=device)
        self.G = nn.Parameter(torch.tensor(G, device=device))
        self._obtain_initial_parameters_by_lr()
        self.save_simulated_parameters()
        self.model = DynamicLinearModel().to(device)

    def generate_simulated_Y(self) -> torch.Tensor:
        """
        Apply the Dynamic Linear Model (DLM) to predict values.

        Returns:
        torch.Tensor: Predicted values.
        """
        self.model.G = self.G
        self.model.eta = self.eta
        self.model.zeta = self.zeta
        self.model.gamma = self.gamma
        simulated_Y, _, _, _, _ = self.model.forward(
            self.X_t, self.Z_t, is_simulation=True
        )
        return simulated_Y

    def get_simulation_results(self) -> dict:
        """
        Get a dictionary with actual and predicted Y values for visualization.

        Returns:
        dict: Dictionary containing actual and predicted Y values.
        """
        simulated_Y = self.generate_simulated_Y().detach().cpu().numpy()
        results_Y = {"actual_Y": self.Y_t.cpu().numpy(), "simulated_Y": simulated_Y}
        utils.log_parameters_results(self.G, self.eta, self.zeta)
        return results_Y
    
    def save_simulated_parameters(self):
        """
        Save the simulated parameters to a CSV file by adding new rows above the original file.
        """
        simulated_params = {
            "Param": ["G"]
            + [f"η{i+1}" for i in range(len(self.eta))]
            + [f"ζ{i+1}" for i in range(len(self.zeta))]
            + [f"γ{i+1}" for i in range(len(self.gamma))],
            "Simulated": [self.G.data.cpu().numpy()]
            + list(self.eta.data.cpu().numpy())
            + list(self.zeta.data.cpu().numpy())
            + list(self.gamma.data.cpu().numpy()),
        }
        df_simulated = pd.DataFrame(simulated_params)

        file_path = config["simulationRecovery"]["simulatedParamSavedPath"]

        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            combined_df = pd.concat([df_simulated, existing_df], ignore_index=True)
        else:
            combined_df = df_simulated

        combined_df.to_csv(file_path, index=False)

    def _obtain_initial_parameters_by_lr(self):
        """
        Obtain eta (η), zeta (ζ) gamma (γ) by fitting a linear regression model to XZ_t and Y_t.
        """
        XZ_t = np.hstack((self.X_t.cpu().numpy(), self.Z_t.cpu().numpy()))
        model_XZ = LinearRegression().fit(XZ_t, self.Y_t.cpu().numpy())
        eta_zeta = model_XZ.coef_

        self.eta = nn.Parameter(
            torch.tensor(eta_zeta[: self.X_t.shape[1]], device=device)
        )
        self.zeta = nn.Parameter(
            torch.tensor(eta_zeta[self.X_t.shape[1] :], device=device)
        )
        # during simulation, we take gamma=zeta
        # self.gamma = self.zeta
        self.gamma = nn.Parameter(torch.randn(config["dataset"]["zDim"], device=device))