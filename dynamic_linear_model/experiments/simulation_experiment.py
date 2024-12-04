import os
import pandas as pd
import torch
import timeit
import matplotlib.pyplot as plt
from loguru import logger

from config import config
from dynamic_linear_model.data_simulation import SimulationRecovery
from dynamic_linear_model.utils import (
    Plotter,
    delete_existing_files,
    generate_combined_simulated_params,
    select_best_performance_simulated_params,
)


def trainer(X_t, Z_t, Y_t, plot_result=True, ax=None):
    """
    Parameters:
    X_t (np.ndarray): Independent variables matrix X.
    Z_t (np.ndarray): Independent variables matrix Z.
    Y_t (np.ndarray): Dependent variable vector Y.
    plot_result(bool): Choose to plot the results or not.
    ax (matplotlib.axes._subplots.AxesSubplot): Subplot to plot results on.

    Returns:
    gamma_best (list): List of gamma best values.
    zeta_best (list): List of zeta best values.
    labels (list): List of labels for zeta parameters.
    """
    start_time = timeit.default_timer()
    delete_existing_files(
        config["simulationRecovery"]["paramsSavedPath"],
        config["simulationRecovery"]["simulatedParamSavedPath"],
    )

    plotter = Plotter()
    val_indices = None
    if config["modelTraining"]["ifCrossValidation"]:
        predicted_Y, val_indices = SimulationRecovery(X_t, Z_t, Y_t).cross_validate(k_folds=5)
    else:
        predicted_Y, val_indices = SimulationRecovery(X_t, Z_t, Y_t).recovery_for_simulation(ax)

    if plot_result:
        plotter.plot(
            config["dataset"]["year_week"][val_indices],
            Y_t[val_indices],
            predicted_Y,
            "Actual Y_t",
            "Predicted Y_t",
            f"{config['dataset']['brand']} - Actual vs Predicted",
            ax
        )
        plotter.setup_and_legend(
            ax, "Time", "Volume Sold", f"{config['dataset']['brand']} - Actual vs Predicted"
        )

    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    logger.info(f"Execution time: {execution_time:.6f} seconds")

    # Get gamma and zeta best values
    gamma_best, zeta_best, labels = plotter.get_gamma_zeta_best_values(config["simulationRecovery"]["paramsSavedPath"])
    return gamma_best, zeta_best, labels


def trainer_with_multi_independent_runs_list(X_t, Z_t, Y_t):

    directory_path = os.path.dirname(config["simulationRecovery"]["paramsSavedPath"])

    for num_runs in config["simulationRecovery"]["independentRunList"]:
        config["simulationRecovery"]["independentRun"] = num_runs
        config["simulationRecovery"]["paramsSavedPath"] = os.path.join(
            directory_path, f"simulated_parameters_{num_runs}.csv"
        )

        trainer(X_t, Z_t, Y_t, plot_result=False)

    generate_combined_simulated_params(directory_path)


def trainer_with_multi_independent_runs(X_t, Z_t, Y_t):

    directory_path = os.path.dirname(config["simulationRecovery"]["paramsSavedPath"])
    num_independet_runs = max(config["simulationRecovery"]["independentRunList"])
    config["simulationRecovery"]["independentRun"] = num_independet_runs

    config["simulationRecovery"]["paramsSavedPath"] = os.path.join(
        directory_path, f"simulated_parameters_total_{num_independet_runs}.csv"
    )
    trainer(X_t, Z_t, Y_t, plot_result=False)

    select_best_performance_simulated_params(
        config["simulationRecovery"]["paramsSavedPath"],
        config["modelTraining"]["lossPath"],
    )
