import os
import torch
import timeit
import matplotlib.pyplot as plt
from loguru import logger

from config import config
from dynamic_linear_model.data_simulation import DataSimulation, SimulationRecovery
from dynamic_linear_model.utils import (
    Plotter,
    delete_existing_files,
    generate_combined_simulated_params,
    select_best_performance_simulated_params,
)


def simulation_recovery(X_t, Z_t, Y_t, plot_result=True):
    """

    Parameters:
    X_t (np.ndarray): Independent variables matrix X.
    Z_t (np.ndarray): Independent variables matrix Z.
    Y_t (np.ndarray): Dependent variable vector Y.
    plot_result(bool): choose to plot the results or not
    """
    start_time = timeit.default_timer()
    delete_existing_files(
        config["simulationRecovery"]["paramsSavedPath"],
        config["simulationRecovery"]["simulatedParamSavedPath"],
    )

    G_list = config["simulationRecovery"]["ListG"]
    num_runs = config["simulationRecovery"]["independentRun"]

    fig, axs = plt.subplots(len(G_list), 2, figsize=(16, 8))
    fig_optim_g, axs_optim_g = plt.subplots(len(G_list), num_runs, figsize=(16, 8))
    fig_training, axs_training = plt.subplots(len(G_list), num_runs, figsize=(16, 8))

    # Ensure axs, axs_optim_g, and axs_training are 2D arrays if G_list has more than one element
    if len(G_list) == 1:
        axs = [axs]
        axs_optim_g = [axs_optim_g]
        axs_training = [axs_training]

    plotter = Plotter()
    for G, ax, ax_training, ax_optim_g in zip(G_list, axs, axs_training, axs_optim_g):
        config["modelTraining"]["originalG"] = G
        logger.info(f"Running on G = {G}")

        data_simulation = DataSimulation(X_t, Z_t, Y_t, G)
        simulated_results = data_simulation.get_simulation_results()
        simulated_Y = torch.tensor(
            simulated_results["simulated_Y"], dtype=torch.float32
        )
        predicted_Y = SimulationRecovery(X_t, Z_t, simulated_Y).recovery_for_simulation(
            ax[0], ax_training, ax_optim_g
        )

        print("simulated_Ysimulated_Y", simulated_Y)
        print("predicted_Y", predicted_Y)
        if plot_result:
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
                "Parameters Value",
                "Change of parameters with different runs",
            )
            plotter.setup_and_legend(
                fig.axes[1], "Time", "Volume Sold", "Simulated vs Predicted"
            )
            plotter.add_config_text(fig)

            fig.tight_layout()
            fig_optim_g.tight_layout()
            fig_training.tight_layout()

    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    logger.info(f"Execution time: {execution_time:.6f} seconds")

    if plot_result:
        plt.tight_layout()
        plt.show()


def simulation_recovery_with_multi_independent_runs_list(X_t, Z_t, Y_t):

    directory_path = os.path.dirname(config["simulationRecovery"]["paramsSavedPath"])

    for num_runs in config["simulationRecovery"]["independentRunList"]:
        config["simulationRecovery"]["independentRun"] = num_runs
        config["simulationRecovery"]["paramsSavedPath"] = os.path.join(
            directory_path, f"simulated_parameters_{num_runs}.csv"
        )

        simulation_recovery(X_t, Z_t, Y_t, plot_result=False)

    generate_combined_simulated_params(directory_path)


def simulation_recovery_with_multi_independent_runs(X_t, Z_t, Y_t):

    directory_path = os.path.dirname(config["simulationRecovery"]["paramsSavedPath"])
    num_independet_runs = max(config["simulationRecovery"]["independentRunList"])
    config["simulationRecovery"]["independentRun"] = num_independet_runs

    config["simulationRecovery"]["paramsSavedPath"] = os.path.join(
        directory_path, f"simulated_parameters_total_{num_independet_runs}.csv"
    )
    simulation_recovery(X_t, Z_t, Y_t, plot_result=False)

    select_best_performance_simulated_params(
        config["simulationRecovery"]["paramsSavedPath"],
        config["modelTraining"]["lossPath"],
    )
