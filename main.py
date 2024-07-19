import logging
from loguru import logger
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import config
from dynamic_linear_model.data_processing import DataPreprocessing
from dynamic_linear_model.data_simulation import DataSimulation, SimulationRecovery
from dynamic_linear_model.utils import Plotter, delete_existing_files

# Configure logging
logging.basicConfig(
    filename="log.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def simulation_recovery(X_t, Z_t, Y_t):
    """

    Parameters:
    X_t (np.ndarray): Independent variables matrix X.
    Z_t (np.ndarray): Independent variables matrix Z.
    Y_t (np.ndarray): Dependent variable vector Y.
    """
    delete_existing_files(
        config["simulationRecovery"]["paramsSavedPath"],
        config["simulationRecovery"]["simulatedParamSavedPath"],
    )

    G_list = config["simulationRecovery"]["ListG"]
    num_runs = config["simulationRecovery"]["independentRun"]
    fig, axs = plt.subplots(len(G_list), 2, figsize=(16, 8))
    fig_optim_g, axs_optim_g = plt.subplots(
        len(G_list), num_runs, figsize=(16, 8)
    )
    fig_training, axs_training = plt.subplots(
        len(G_list), num_runs, figsize=(16, 8)
    )

    # Ensure axs, axs_optim_g, and axs_training are 2D arrays if G_list has more than one element
    if len(G_list) == 1:
        axs = [axs]
        axs_optim_g = [axs_optim_g]
        axs_training = [axs_training]

    plotter = Plotter()
    for G, ax, ax_training, ax_optim_g in zip(G_list, axs, axs_training, axs_optim_g):
        config["modelTraining"]["originalG"] = G

        data_simulation = DataSimulation(X_t, Z_t, Y_t, G)
        simulated_results = data_simulation.get_simulation_results()
        simulated_Y = torch.tensor(
            simulated_results["simulated_Y"], dtype=torch.float32
        )
        predicted_Y = SimulationRecovery(X_t, Z_t, simulated_Y).recovery_for_simulation(
            ax[0], ax_training, ax_optim_g
        )

        plotter.plot(
            simulated_Y,
            predicted_Y,
            "Simulated Y_t",
            "Predicted Y_t",
            "Simulated vs Predicted",
            ax[1],
        )

        plotter.setup_and_legend(
            fig_optim_g, "Epoch", "value", "change of g value during optimizing"
        )
        plotter.setup_and_legend(
            fig_training, "Epoch", "Loss", "Change of Loss during Recovery"
        )
        plotter.setup_and_legend(
            fig.axes[0],
            "Independent Run",
            "Loss",
            "Change of parameters with different run",
        )
        plotter.setup_and_legend(
            fig.axes[1], "Time", "Volume Sold", "Simulated vs Predicted"
        )
        plotter.add_config_text(fig)


if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(
        config["dataset"]["path"],
        config["dataset"]["brand"],
        config["dataset"]["dependent_variable"],
        config["dataset"]["independent_variables_X"],
        config["dataset"]["independent_variables_Z"],
    )

    X_t, Z_t, Y_t = data_preprocessing.preprocess(normalization=True)
    import time

    start = time.time()
    simulation_recovery(X_t, Z_t, Y_t)
    end = time.time()
    logger.info(f"total running time: {end - start}")

    plt.tight_layout()
    plt.show()
