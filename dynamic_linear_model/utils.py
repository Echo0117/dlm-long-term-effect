import glob
import logging
import os
import random
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from config import config
from adjustText import adjust_text


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
        logger.info(f"File {file_path} has been deleted successfully.")
    elif os.path.exists(file_simulated_path):
        os.remove(file_simulated_path)
        logger.info(f"File {file_simulated_path} has been deleted successfully.")
    else:
        logger.info(f"Files {file_path} and {file_simulated_path} do not exist.")


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
    recovered_parameters, best_run_number: int
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
        recovered_parameters["G"],
        recovered_parameters["eta"],
        recovered_parameters["zeta"],
        recovered_parameters["gamma"],
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
        + ["Mean", "Std", "Median", "Best"]
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

    # df["Simulated"] = df_simulated_existing["Simulated"][: len(rows)]
    new_df = pd.concat([df, df_existing], axis=0)
    empty_row = pd.DataFrame([[None] * len(new_df.columns)], columns=new_df.columns)
    new_df = pd.concat([empty_row, new_df], ignore_index=True)

    # Save to CSV
    saved_file_path = config["simulationRecovery"]["paramsSavedPath"]
    print("saved_file_path", saved_file_path)
    new_df.to_csv(saved_file_path, index=False)

    logger.info(f"Simulation results saved to {saved_file_path}")


def generate_combined_simulated_params(directory):
    # Find all files with the prefix 'simulated_parameters_'
    files = glob.glob(f'{directory}/simulated_parameters_*.csv')
    # Sort files by numerical order of their postfix
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    data_dict = {}

    # Extract the relevant columns from each file
    for idx, file in enumerate(files):
        df = pd.read_csv(file)
        postfix = file.split('_')[-1].split('.')[0]  # Extract the postfix (e.g., '2' from 'simulated_parameters_2.csv')
        column_name = f'independent run {postfix}'
        data_dict[column_name] = df[['Best', 'Mean', 'Std']]

        if idx == 0:
            simulated_column = df['Simulated']

    combined_df = pd.concat(data_dict, axis=1)

    # Flatten the MultiIndex in columns
    combined_df.columns = [f'{col[1]} ({col[0]})' for col in combined_df.columns]

    combined_df.insert(0, 'Parameter', df['Param'])
    combined_df.insert(1, 'Simulated', simulated_column)

    output_file = f'{directory}/combined_simulated_parameters.csv'
    combined_df.to_csv(output_file, index=False)


def select_best_performance_simulated_params(simulated_parameters_path, loss_results_path):

    simulated_parameters_df = pd.read_csv(simulated_parameters_path)
    loss_results_df = pd.read_csv(loss_results_path)
    combined_simulated_parameters_df = simulated_parameters_df.copy()

    for iteration in config["simulationRecovery"]["independentRunList"]:
        # Assuming df and loss_results_df are defined DataFrames
        iteration_columns = [col for col in simulated_parameters_df.columns if col.startswith('iteration')]

        selected_columns = random.sample(iteration_columns, iteration)
        selected_columns_index = [int(s.split()[-1])-1 for s in selected_columns]

        selected_losses = loss_results_df.iloc[selected_columns_index].reset_index() # Row 1 contains the loss values
        min_loss_row = selected_losses.iloc[selected_losses['loss value'].idxmin()]

        # Extract the independent run value from the row
        independent_run_value = int(min_loss_row['independent run'])

        # Selecting iteration 1 and iteration 2 columns
        selected_iterations = simulated_parameters_df[selected_columns]

        # Calculating mean and std for each row
        combined_simulated_parameters_df[f'Mean (independent run {iteration})'] = selected_iterations.mean(axis=1)
        combined_simulated_parameters_df[f'Std (independent run {iteration})'] = selected_iterations.std(axis=1)
        combined_simulated_parameters_df[f'Best (independent run {iteration})'] = selected_iterations[f"iteration {independent_run_value}"]
        
    output_file = f'{os.path.dirname(config["simulationRecovery"]["paramsSavedPath"])}/combined_simulated_parameters.csv'
    combined_simulated_parameters_df.to_csv(output_file, index=False)

class Plotter:
    def plot(
        self, dates, data_1, data_2, label1: str, label2: str, title: str, ax: Axes
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
        dates_with_day = dates.astype(str) + '1'
        date_series = pd.to_datetime(dates_with_day, format='%Y%W%w')
        ax.plot(date_series, data_1, "b-", label=label1)
        ax.plot(date_series, data_2, "r--", label=label2)

        # Find peak values and annotate them
        self._annotate_peaks(ax, date_series, data_1, label1, color='blue')
        self._annotate_peaks(ax, date_series, data_2, label2, color='red')
        
        # Format the x-axis to show months
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title(
            f'{title} Simulated G = {"{:.1f}".format(config["modelTraining"]["originalG"])}'
        )

    def _annotate_peaks(self, ax, dates, data, label, color, num_peaks=5):
        """
        Annotate the top 'num_peaks' peak values in the data.
        """
        # Find indices of the top 'num_peaks' maximum values
        peak_indices = np.argsort(data)[-num_peaks:]
        
        # Sort peak indices to plot in chronological order
        peak_indices = np.sort(peak_indices)
        
        # Offset for annotations to avoid overlapping
        offsets = np.linspace(15, 75, num_peaks)
        
        for i, index in enumerate(peak_indices):
            peak_value = data[index]
            peak_date = dates[index]
            peak_date_str = peak_date.strftime('%b %Y')
            
            # Adjust annotation position with offsets
            xytext = (0, offsets[i])
            
            # Annotate each peak
            ax.annotate(
                f'{label} Peak:\n{peak_value:.2f} ({peak_date_str})',
                xy=(peak_date, peak_value),
                xytext=xytext,
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=8,
                color=color,
                arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=5),
            )

    def plot_metrics_multiprocess(
        self, metrics: tuple, ax_training: list, ax_optim_g: list
    ) -> None:
        """
        Plot losses and parameter changes from collected metrics.

        Parameters:
        metrics (tuple): List of metrics from each run.
        ax_training (list): List of matplotlib axes for training metrics.
        ax_optim_g (list): List of matplotlib axes for optimization metrics.
        """
        start_epoch = 10

        # Ensure ax_training and ax_optim_g are lists of axes
        if isinstance(ax_training, Axes):
            ax_training = [ax_training]
        if isinstance(ax_optim_g, Axes):
            ax_optim_g = [ax_optim_g]

        params_before_list, params_after_optim_list, losses_list = metrics
        
        for i, (
            params_before, params_after_optim, losses,
            ax_training_sub,
            ax_optim_g_sub,
        ) in enumerate(zip(params_before_list, params_after_optim_list, losses_list, ax_training, ax_optim_g)):
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
            ax_training_sub.set_title(
            f'Independent run {i} with G = {"{:.1f}".format(config["modelTraining"]["originalG"])}',
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
            ax_optim_g_sub.set_title(
            f'Independent run {i} with G = {"{:.1f}".format(config["modelTraining"]["originalG"])}',
            )

    def plot_params(self, params, ax: Axes) -> tuple:
        """
        Plot the G, eta, zeta, and gamma values for each run.
        """
        num_runs = config["simulationRecovery"]["independentRun"]
       
        parameters = {"G": None, "eta": None, "zeta": None, "gamma": None}

        # for _, params in enumerate(params_list):
        for name, param in params:
            if name in parameters:
                if name == "G":
                    parameters[name] = torch.sigmoid(param).data.cpu().numpy()
                else:
                    parameters[name] = param.data.cpu().numpy()
        
        # ax.plot(range(num_runs), parameters["G"], label=f"G")
        # parameters_plotting = {
        #     "eta": parameters["eta"].T,
        #     "zeta": parameters["zeta"].T,
        #     "gamma": parameters["gamma"].T,
        # }
        # for param_name, param_values in parameters_plotting.items():
        #     for i in range(param_values.shape[0])[:2]:
        #         ax.plot(range(num_runs), param_values[i], label=f"{param_name} {i+1}")
        # ax.set_xticks(range(num_runs))

        return parameters

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


    def plot_gamma_zeta_best_values(self, dataset_path, independent_variables_Z=config["dataset"]["independent_variables_Z"]):
        """
        Generates a scatter plot of best values for gamma and zeta parameters from a specified dataset.
        
        Args:
            dataset_path (str): Path to the CSV file containing the dataset.
            independent_variables_Z (list): List of independent variable labels for zeta parameters.
        """
        # Check if the dataset file exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The dataset file was not found at the path: {dataset_path}")

        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Define the list of gamma and zeta parameters
        gamma_params = [f'γ{i}' for i in range(1, 10)]
        zeta_params = [f'ζ{i}' for i in range(1, 10)]

        # Check if all required gamma and zeta parameters are present
        required_gamma_zeta = gamma_params + zeta_params
        missing_params = set(required_gamma_zeta) - set(df['Param'])
        if missing_params:
            raise ValueError(f"The following required parameters are missing in the dataset: {missing_params}")

        # Extract Best values for gamma and zeta
        gamma_best = df[df['Param'].isin(gamma_params)][['Param', 'Best']].reset_index(drop=True)
        zeta_best = df[df['Param'].isin(zeta_params)][['Param', 'Best']].reset_index(drop=True)

        # Ensure that the number of zeta parameters matches independent_variables_Z
        if len(zeta_best) != len(independent_variables_Z):
            raise ValueError("The number of zeta parameters does not match the number of independent_variables_Z.")

        # Sort gamma and zeta to ensure correct pairing (optional)
        gamma_best = gamma_best.sort_values(by='Param').reset_index(drop=True)
        zeta_best = zeta_best.sort_values(by='Param').reset_index(drop=True)

        # Map zeta parameters to independent_variables_Z
        zeta_best['Label'] = independent_variables_Z

        # Combine gamma and zeta Best values into a single DataFrame
        combined_df = pd.DataFrame({
            'Gamma_Best': gamma_best['Best'],
            'Zeta_Best': zeta_best['Best'],
            'Label': zeta_best['Label']
        })

        # Create the scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            combined_df['Gamma_Best'],
            combined_df['Zeta_Best'],
            c=combined_df['Gamma_Best'],
            cmap='viridis',
            s=150,
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )

        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Gamma Best')

        # Prepare text labels
        texts = [
            plt.text(row['Gamma_Best'] + 0.02, row['Zeta_Best'] + 0.02, row['Label'],
                    fontsize=9, ha='left', va='bottom')
            for idx, row in combined_df.iterrows()
        ]

        # Adjust text to minimize overlaps
        adjust_text(
            texts,
            arrowprops=dict(arrowstyle='->', color='red', linewidth=0.5),
            expand_points=(1.2, 1.2),
            force_points=0.2,
            force_text=0.2,
            lim=1000
        )

        # Set labels and title
        plt.xlabel('Gamma Best')
        plt.ylabel('Zeta Best')
        plt.title(f"""{config["dataset"]["brand"]} - Scatter Plot of Best Values: Gamma vs. Zeta """)

        # Enable grid
        plt.grid(True, linestyle='--', alpha=0.5)

        # Show the plot
        plt.tight_layout()

    def get_gamma_zeta_best_values(self, filepath):
        """
        Reads gamma and zeta best values along with labels from the given CSV file.

        Parameters:
        filepath (str): Path to the CSV file.

        Returns:
        gamma_best (list): List of gamma best values.
        zeta_best (list): List of zeta best values.
        labels (list): List of labels for zeta parameters.
        """
        df = pd.read_csv(filepath)
        gamma_params = [f'γ{i}' for i in range(1, 10)]
        zeta_params = [f'ζ{i}' for i in range(1, 10)]

        gamma_best = df[df['Param'].isin(gamma_params)]['Best'].tolist()
        zeta_best = df[df['Param'].isin(zeta_params)]['Best'].tolist()
        # labels = df[df['Param'].isin(zeta_params)]['Param'].tolist()  # Labels correspond to zeta parameters
        labels = config["dataset"]["independent_variables_Z"]
        return gamma_best, zeta_best, labels


def plot_gamma_zeta_combined(gamma_best, zeta_best, labels, brands, ax):
    """
    Plots gamma vs. zeta best values for all brands on a single plot.

    Parameters:
    gamma_best (list): List of gamma best values.
    zeta_best (list): List of zeta best values.
    labels (list): List of labels for zeta parameters.
    brands (list): List of brand names.
    ax (matplotlib.axes._subplots.AxesSubplot): The axis to plot on.
    """
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_color_map = {label: color for label, color in zip(unique_labels, colors)}

    for gamma, zeta, label, brand in zip(gamma_best, zeta_best, labels, brands):
        color = label_color_map[label]
        ax.scatter(gamma, zeta, color=color, label=label if label not in ax.get_legend_handles_labels()[1] else "", s=100, alpha=0.7, edgecolors='w', linewidth=0.5)
        ax.text(gamma + 0.01, zeta + 0.01, f"{label} ({brand})", fontsize=8, color=color)

    ax.set_xlabel("Gamma Best")
    ax.set_ylabel("Zeta Best")
    ax.set_title("Combined Plot of Gamma vs. Zeta Best Values Across Brands")
    ax.legend(title="Independent Variables (Colors)")
    ax.grid(True, linestyle='--', alpha=0.4)

