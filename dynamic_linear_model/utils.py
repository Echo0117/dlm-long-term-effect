import logging
import os
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import config


def convert_normalized_data(scaler, data):
    convert_data = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    return convert_data


def delete_existing_files(file_path, file_simulated_path):
    """
    Check if the specified files exist and delete them if they do.

    Parameters:
    file_path (str): Path to the first file to be checked and deleted.
    file_simulated_path (str): Path to the second file to be checked and deleted.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} has been deleted successfully.")
    elif os.path.exists(file_simulated_path):
        os.remove(file_simulated_path)
        print(f"File {file_simulated_path} has been deleted successfully.")
    else:
        print(f"Files {file_path} and {file_simulated_path} do not exist.")


def log_parameters_results(
    simulated_G=None, simulated_eta=None, simulated_zeta=None, params=None
):
    """
    Log the simulation results to a log file.

    Parameters:
    simulated_G (float): The G parameter from the simulation.
    simulated_eta (np.ndarray): The eta parameter from the simulation.
    simulated_zeta (np.ndarray): The zeta parameter from the simulation.
    """
    logging.info("Simulation Results:")
    if params:
        logging.info(f"params: {params}")
    else:
        logging.info(f"Simulated G: {simulated_G}")
        logging.info(f"Simulated eta: {simulated_eta}")
        logging.info(f"Simulated zeta: {simulated_zeta}")


def calculate_statistics(
    Gs: list, etas: list, zetas: list, gammas: list, best_run_number: int
) -> None:
    """
    Calculate and print the statistics of the parameters from multiple runs.

    Parameters:
    Gs (list): List of G values.
    etas (list): List of eta values.
    zetas (list): List of zeta values.
    gammas (list): List of gamma values.
    """
    Gs, etas, zetas, gammas = (
        np.array(Gs),
        np.array(etas),
        np.array(zetas),
        np.array(gammas),
    )

    def compute_stats(data, axis=None):
        return {
            "mean": np.mean(data, axis=axis),
            "std": np.std(data, axis=axis),
            "median": np.median(data, axis=axis),
        }

    G_stats = compute_stats(Gs)
    eta_stats = compute_stats(etas, axis=0)
    zeta_stats = compute_stats(zetas, axis=0)
    gamma_stats = compute_stats(gammas, axis=0)

    # Combine all statistics and iteration results into a single DataFrame
    rows = [
        [
            "G",
            *Gs.tolist(),
            G_stats["mean"],
            G_stats["std"],
            G_stats["median"],
            Gs[best_run_number],  # Add best_loss for G
        ]
    ]

    def add_rows(param_name, param_data, param_stats):
        for i in range(len(param_stats["mean"])):
            rows.append(
                [
                    f"{param_name}{i+1}",
                    *param_data[:, i].tolist(),
                    param_stats["mean"][i],
                    param_stats["std"][i],
                    param_stats["median"][i],
                    param_data[:, i][
                        best_run_number
                    ],  # Add best_loss for each parameter
                ]
            )

    add_rows("η", etas, eta_stats)
    add_rows("ζ", zetas, zeta_stats)
    add_rows("γ", gammas, gamma_stats)

    columns = (
        ["Param"]
        + [f"iteration {i+1}" for i in range(len(Gs))]
        + ["Mean", "Std", "Median", "Best Loss"]
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
    new_df = pd.concat([empty_row, new_df], ignore_index=True)

    # Save to CSV
    new_df.to_csv(config["simulationRecovery"]["paramsSavedPath"], index=False)

    print("Simulation results saved to simulation_results.csv")


class Plotter:
    def plot(
        self, data_1, data_2, label1: str, label2: str, title: str, ax: Axes
    ) -> None:
        """
        Plot two datasets with given labels and title.

        Parameters:
        data_1 (array-like): First dataset to plot.
        data_2 (array-like): Second dataset to plot.
        label1 (str): Label for the first dataset.
        label2 (str): Label for the second dataset.
        title (str): Title of the plot.
        ax (Axes): Matplotlib Axes object to plot on.
        """
        ax.plot(data_1, "b-", label=label1)
        ax.plot(data_2, "r--", label=label2)

        original_g = config["modelTraining"]["originalG"]
        config_text = ""
        if config["inferenceMethod"] == "mcmc":
            mcmc_params = config["inferenceParams"]["mcmc"]
            config_text += "inferenceParams: "
            for key, value in mcmc_params.items():
                config_text += f"  {key}: {value}\n"

        config_text += f"originalG: {original_g}"

        ax.set_title(
            f'{title} Simulated G = {"{:.1f}".format(config["modelTraining"]["originalG"])}'
        )

    def plot_metrics_multiprocess(
        self, metrics_list: list, ax_training: list, ax_optim_g: list
    ) -> None:
        """
        Plot losses and parameter changes from collected metrics.

        Parameters:
        metrics_list (list): List of metrics from each run.
        ax_training (list): List of matplotlib axes for training metrics.
        ax_optim_g (list): List of matplotlib axes for optimization metrics.
        """
        start_epoch = 10

        # Ensure ax_training and ax_optim_g are lists of axes
        if not isinstance(ax_training, list):
            ax_training = [ax_training]
        if not isinstance(ax_optim_g, list):
            ax_optim_g = [ax_optim_g]

        for (
            (params_before, params_after_optim, losses),
            ax_training_sub,
            ax_optim_g_sub,
        ) in zip(metrics_list, ax_training, ax_optim_g):
            ax_training_sub.plot(
                range(start_epoch, config["modelTraining"]["epoch"]),
                losses[start_epoch:],
                label="Loss",
            )

            # Annotate the last value of losses
            last_epoch = config["modelTraining"]["epoch"] - 1
            last_loss_value = losses[-1]
            ax_training_sub.annotate(
                f'{last_loss_value:.2f}',
                xy=(last_epoch, last_loss_value),
                xytext=(last_epoch + 1, last_loss_value),
                arrowprops=dict(facecolor='black', shrink=0.05),
            )

            ax_optim_g_sub.plot(
                range(len(params_before)),
                params_before,
                label=f"G Before Optimization ",
            )

            ax_optim_g_sub.plot(
                range(len(params_after_optim)),
                params_after_optim,
                label=f"G After Optimization",
            )

            # Annotate the last value of Optimized G
            last_optimized_g_value = params_after_optim[-1]
            ax_optim_g_sub.annotate(
                f'{last_optimized_g_value:.2f}',
                xy=(last_epoch, last_optimized_g_value),
                xytext=(last_epoch + 1, last_optimized_g_value),
                arrowprops=dict(facecolor='black', shrink=0.05),
            )

    def plot_params(self, params_list: list, ax: Axes) -> tuple:
        """
        Plot the G, eta, zeta, and gamma values for each run.

        Parameters:
        params_list (list): List of parameters from each run.
        ax (matplotlib.axes.Axes): Axes for plotting.

        Returns:
        tuple: Tuple containing lists of G, eta, zeta, and gamma values.
        """
        num_runs = len(params_list)

        Gs, etas, zetas, gammas = [], [], [], []
        param_mappings = {"G": Gs, "eta": etas, "zeta": zetas, "gamma": gammas}

        for _, params in enumerate(params_list):
            for name, param in params:
                if name in param_mappings:
                    if name == "G":
                        param_mappings[name].append(
                            torch.sigmoid(param).data.cpu().numpy()
                        )
                    else:
                        param_mappings[name].append(param.data.cpu().numpy())

        ax.plot(range(num_runs), Gs, label=f"G")
        parameters = {
            "eta": np.array(etas).T,
            "zeta": np.array(zetas).T,
            "gamma": np.array(gammas).T,
        }
        for param_name, param_values in parameters.items():
            for i in range(param_values.shape[0])[:2]:
                ax.plot(range(num_runs), param_values[i], label=f"{param_name} {i+1}")

        return Gs, etas, zetas, gammas

    def setup_and_legend(
        self,
        fig,
        xlabel: str,
        ylabel: str,
        title: str,
        legend_loc: str = "upper right",
        ncol: int = 2,
    ) -> None:
        """
        Setup plot labels, title and legend.

        Parameters:
        fig (Figure or Axes): Matplotlib Figure or Axes object to setup.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        title (str): Title of the plot.
        legend_loc (str, optional): Location of the legend. Defaults to "upper right".
        ncol (int, optional): Number of columns in the legend. Defaults to 2.
        """
        handles, labels = [], []
        if isinstance(fig, Figure):
            fig.supxlabel(xlabel)
            fig.supylabel(ylabel)
            fig.suptitle(title)
            for handle in fig.axes[0].get_lines():
                handles.append(handle)
                labels.append(handle.get_label())
            fig.tight_layout()

        elif isinstance(fig, Axes):
            fig.set_xlabel(xlabel)
            fig.set_ylabel(ylabel)
            fig.set_title(title)
            for handle in fig.get_lines():
                handles.append(handle)
                labels.append(handle.get_label())
        fig.legend(handles, labels, loc=legend_loc, ncol=ncol)

    def add_config_text(self, fig: Figure) -> None:
        """
        Add configuration text to the figure.

        Parameters:
        fig (Figure): The figure to add the text to.
        """
        config_text = ""
        if config["inferenceMethod"] == "torch_autograd":
            gd_params = config["modelTraining"]
            config_text += "inferenceParams: "
            for key, value in gd_params.items():
                if key not in ["modelPath", "nSplits"]:
                    config_text += f"  {key}: {value}\n"
        fig.text(0.05, 0.1, config_text, fontsize=7, va="top", wrap=True)

        original_g = config["modelTraining"]["originalG"]
        config_text = ""
        if config["inferenceMethod"] == "mcmc":
            mcmc_params = config["inferenceParams"]["mcmc"]
            config_text += "inferenceParams: "
            for key, value in mcmc_params.items():
                config_text += f"  {key}: {value}\n"

        config_text += f"originalG: {original_g}"
        fig.text(0.05, 0.05, config_text, fontsize=7, va="top", wrap=True)

    def plot_metrics_singleprocess(self, losses: list, ax: Axes, num_run: int) -> None:
        """
        Plot training loss.

        Parameters:
        losses (list): List of losses.
        ax (matplotlib.axes.Axes): Axes for plotting.
        num_run (int): The current run number.
        """
        ax.plot(range(self.epoch), losses, "b-", label="Training loss")
        ax.set_title(
            f'Independent run {num_run} with G = {"{:.1f}".format(config["modelTraining"]["originalG"])}',
        )