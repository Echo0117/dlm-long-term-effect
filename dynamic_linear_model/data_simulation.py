import logging
import os
import pdb
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch.multiprocessing as mp
import ipdb
from config import config
from dynamic_linear_model.losses import mse_loss
from dynamic_linear_model.model import DynamicLinearModel
import dynamic_linear_model.utils as utils


device = config["device"]

from config import config
from dynamic_linear_model.losses import mse_loss
from dynamic_linear_model.model import DynamicLinearModel
import dynamic_linear_model.utils as utils
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import logging
from matplotlib.axes import Axes

device = config["device"]


class SimulationRecovery:
    def __init__(self, X_t: np.ndarray, Z_t: np.ndarray, Y_t: np.ndarray):
        self.X_t = torch.tensor(X_t, dtype=torch.float32, device=device)
        self.Z_t = torch.tensor(Z_t, dtype=torch.float32, device=device)
        self.Y_t = torch.tensor(Y_t, dtype=torch.float32, device=device)
        self.epoch = config["modelTraining"]["epoch"]
        self.num_runs = config["simulationRecovery"]["independentRun"]

        # Initialize the model with parameters for all runs
        self.model = DynamicLinearModel(self.num_runs).to(device)

    def recovery_for_simulation(
        self, ax: Axes, ax_training: Axes, ax_optim_g: Axes
    ) -> np.ndarray:

        plotter = utils.Plotter()

        (
            named_parameters,
            # params_before,
            # params_after_optim,
            # losses,
            best_epoch,
            best_loss_run_index,
            best_loss,
            best_predicted_Y,
        ) = self._optimize(self.Y_t, self.X_t, self.Z_t)

        logger.info(
            f"best final loss: {best_loss}, best final loss is at epoch: {best_epoch}, the index of best run: {best_loss_run_index}"
        )
        # metrics_list = (params_before, params_after_optim, losses)
        # plotter.plot_metrics_multiprocess(metrics_list, ax_training, ax_optim_g)

        recovered_parameters = plotter.plot_params(named_parameters, ax)
        utils.calculate_statistics(recovered_parameters, best_loss_run_index)

        return best_predicted_Y

    def _optimize(
        self, Y_t: torch.Tensor, X_t: torch.Tensor, Z_t: torch.Tensor
    ) -> tuple:
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=config["modelTraining"]["learningRate"],
            momentum=config["modelTraining"]["momentum"],
            weight_decay=config["modelTraining"]["weightDecay"],
        )

        (
            outputs_list,
            losses_list,
            # g_params_before_list,
            # g_params_after_optim_list,
            all_parameters_list,
        ) = ([], [], [])

        for epoch in tqdm(range(self.epoch), desc="Epochs"):

            optimizer.zero_grad()
            # outputs, G_hat, _, _, _ = self.model.forward(X_t, Z_t)
            outputs = self.model.forward(X_t, Z_t)


            # params_before_epoch = G_hat.data
            # params_before_optim = self.model.named_parameters()
            batch_losses = mse_loss(outputs, Y_t.repeat(self.num_runs, 1))

            # Backward each loss separately
            for loss in batch_losses:
                loss.backward(retain_graph=True)
            optimizer.step()

            # g_params_after_optim = [
            #     torch.sigmoid(param.clone().data) for param in self.model.G
            # ]
            params_after_optim = self.model.named_parameters()

            # g_params_before_list.append(G_hat.data)
            # g_params_after_optim_list.append(g_params_after_optim)
            all_parameters_list.append(params_after_optim)
            # losses_list.append(batch_losses.data.cpu().numpy())
            # outputs_list.append(outputs.data.cpu().numpy())
            losses_list.append(batch_losses.detach().clone())
            outputs_list.append(outputs.detach().clone())

            if epoch % 100 == 0:
                logging.info(
                    f"epoch {epoch}: Total mean Loss = {batch_losses.mean().item()}"
                )

        for name, param in self.model.named_parameters():
            logging.info(f"Params {name}: {param.data}")

        # ipdb.set_trace()

        # params_before_np = np.array(g_params_before_list)
        # params_after_optim_np = np.array(g_params_after_optim_list)
        # losses_np = np.array(losses_list)
        # outputs_np = np.array(outputs_list)

        losses_tensor = torch.stack(losses_list)  # Shape: (epoch, num_runs, T)
        outputs_tensor = torch.stack(outputs_list)  # Shape: (epoch, num_runs, T)
        losses_np = losses_tensor.cpu().numpy()  # Move once to CPU and convert
        outputs_np = outputs_tensor.cpu().numpy()


        # Find the index of the index of best performance
        min_index_1d = np.argmin(losses_np)
        min_index_2d = np.unravel_index(min_index_1d, losses_np.shape)

        best_epoch = min_index_2d[0]
        best_loss_run_index = min_index_2d[1]
        best_loss = losses_np[min_index_2d]
        best_predicted_Y = outputs_np[min_index_2d]

        # Create a DataFrame with the required structure
        min_indices_by_column_independent = np.argmin(losses_np, axis=0)
        min_values_by_column_independent = losses_np[
            min_indices_by_column_independent, np.arange(losses_np.shape[1])
        ]
        loss_data = {
            "independent run": [
                x + 1 for x in list(range(len(min_indices_by_column_independent)))
            ],
            "epoch index": min_indices_by_column_independent,
            "loss value": min_values_by_column_independent,
        }
        pd.DataFrame(loss_data).to_csv(config["modelTraining"]["lossPath"])
        logger.info(f"file loss_results saved at {config['modelTraining']['lossPath']}")

        # Convert the generator to a list of dictionaries
        param_list = list(all_parameters_list)

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(param_list)
        df.columns = ["G", "eta", "zeta", "gamma"]
        df.to_csv(config["modelTraining"]["parametersPath"], index=False)

        return (
            self.model.named_parameters(),
            # params_before_np.T,
            # params_after_optim_np.T,
            # losses_np.T,
            best_epoch,
            best_loss_run_index,
            best_loss,
            best_predicted_Y,
        )

    def _get_predicted_Y(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        predicted_Y = self.model(X, Z)
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
        self.model = DynamicLinearModel(num_runs=1).to(device)

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
        # simulated_Y, _, _, _, _ = self.model.forward(
        #     self.X_t, self.Z_t, is_simulation=True
        # )
        simulated_Y = self.model.forward(
            self.X_t, self.Z_t, is_simulation=True
        )
        return simulated_Y[0]

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
        model_XZ = LinearRegression(
            fit_intercept=False,
            # positive=True
        ).fit(XZ_t, self.Y_t.cpu().numpy())
        eta_zeta = model_XZ.coef_

        self.eta = nn.Parameter(
            torch.tensor(eta_zeta[: self.X_t.shape[1]], device=device)
        )
        self.zeta = nn.Parameter(
            torch.tensor(eta_zeta[self.X_t.shape[1] :], device=device)
        )

        if config["simulationRecovery"]["isGammaEqualZeta"]:
            self.gamma = self.zeta
        else:
            self.gamma = nn.Parameter(
                torch.randn(config["dataset"]["zDim"], device=device)
            )
