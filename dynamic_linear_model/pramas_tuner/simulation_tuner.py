import os
import sys
import numpy as np
import torch
import torch.optim as optim
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import report
from loguru import logger
from matplotlib.axes import Axes

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from config import config
from dynamic_linear_model import utils
from dynamic_linear_model.model import DynamicLinearModel
from dynamic_linear_model.data_processing import DataPreprocessing
from dynamic_linear_model.data_simulation import DataSimulation


device = config["device"]


class SimulationRecovery:
    def __init__(self, X_t: np.ndarray, Z_t: np.ndarray, Y_t: np.ndarray):
        self.X_t = torch.tensor(X_t, dtype=torch.float32, device=device)
        self.Z_t = torch.tensor(Z_t, dtype=torch.float32, device=device)
        self.Y_t = torch.tensor(Y_t, dtype=torch.float32, device=device)
        self.epoch = config["modelTraining"]["epoch"]
        self.model = DynamicLinearModel().to(device)

    def run_optimization(self, config, checkpoint_dir=None):
        self.model = DynamicLinearModel().to(device)
        try:
            params_before, params_after_optim, losses = self._optimize(
                self.Y_t,
                self.X_t,
                self.Z_t,
                config
            )
        except (UnboundLocalError, RuntimeError) as e:
            logger.error(e)
            report(dict(loss=float("inf")))
            return

        Y_predicted = self._get_predicted_Y(self.X_t, self.Z_t)
        final_loss = losses[-1]
        report(dict(loss=final_loss))

    def recovery_for_simulation(self, ax: Axes, ax_training: Axes, ax_optim_g: Axes) -> np.ndarray:
        num_runs = config["simulationRecovery"]["independentRun"]
        params_list, metrics_list = [], []
        best_Y_predicted = None
        plotter = utils.Plotter()

        # Define the search space
        search_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "momentum": tune.uniform(0.1, 0.9),
            "weight_decay": tune.loguniform(1e-5, 1e-2)
        }

        # Use ASHA scheduler for early stopping of bad runs
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=self.epoch,
            grace_period=1,
            reduction_factor=2
        )

        # Run the tuning
        analysis = tune.run(
            self.run_optimization,
            config=search_space,
            num_samples=num_runs,
            scheduler=scheduler
        )

        # Get the best result
        best_config = analysis.get_best_config(metric="loss", mode="min")

        # Retrain with the best config
        best_params, best_Y_predicted, best_metrics, best_final_loss = self.run_optimization(best_config)
        params_list.append(best_params)
        metrics_list.append(best_metrics)

        # Plot metrics after multiprocessing
        plotter.plot_metrics_multiprocess(metrics_list, ax_training, ax_optim_g)

        # Calculate statistics
        Gs, etas, zetas, gammas = plotter.plot_params(params_list, ax)
        utils.calculate_statistics(Gs, etas, zetas, gammas, 0)  # Assuming best_run_number = 0

        return best_Y_predicted

    def mse_loss(self, predicted_Y: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.mean((predicted_Y - Y) ** 2)

    def _optimize(self, Y_t: torch.Tensor, X_t: torch.Tensor, Z_t: torch.Tensor, config) -> tuple:
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"]
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

        return params_before, params_after_optim, losses

    def _get_predicted_Y(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        predicted_Y, _, _, _, _ = self.model(X, Z)
        return predicted_Y
    

def simulation_recovery(X_t, Z_t, Y_t):

    G_list = config["simulationRecovery"]["ListG"]

    for G in G_list:
        logger.info(f"G = {G}")
        config["modelTraining"]["originalG"] = G

        data_simulation = DataSimulation(X_t, Z_t, Y_t, G)
        simulated_results = data_simulation.get_simulation_results()
        simulated_Y = torch.tensor(
            simulated_results["simulated_Y"], dtype=torch.float32
        )

        # Using Ray Tune for recovery
        analysis = tune.run(
            lambda config: SimulationRecovery(X_t, Z_t, simulated_Y).run_optimization(config),
            config={
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "momentum": tune.uniform(0.1, 0.9),
                "weight_decay": tune.loguniform(1e-5, 1e-2)
            },
            num_samples=config["simulationRecovery"]["independentRun"],
            scheduler=ASHAScheduler(metric="loss", mode="min", max_t=config["modelTraining"]["epoch"])
        )

        best_config = analysis.get_best_config(metric="loss", mode="min")
        logger.info(f'best config {best_config}')


if __name__ == "__main__":
    ray.init()

    data_preprocessing = DataPreprocessing(
        config["dataset"]["path"],
        config["dataset"]["brand"],
        config["dataset"]["dependent_variable"],
        config["dataset"]["independent_variables_X"],
        config["dataset"]["independent_variables_Z"],
    )

    X_t, Z_t, Y_t = data_preprocessing.preprocess(normalization=True)
    simulation_recovery(X_t, Z_t, Y_t)

    ray.shutdown()
