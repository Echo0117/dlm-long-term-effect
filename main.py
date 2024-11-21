import os
import logging
from matplotlib import pyplot as plt
from config import config 
from dynamic_linear_model.data_processing import DataPreprocessing
from dynamic_linear_model.experiments.simulation_experiment import (
    trainer,
    trainer_with_multi_independent_runs,
    trainer_with_multi_independent_runs_list
)
from dynamic_linear_model.utils import plot_gamma_zeta_combined
import math
import numpy as np

# Configure logging
logging.basicConfig(
    filename="log.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":
    # Determine the number of brands and calculate a suitable grid size (rows and columns)
    num_brands = len(config["dataset"]["brand_list"])
    cols = min(4, num_brands)  # Limit columns to 4 for readability
    rows = math.ceil(num_brands / cols)  # Calculate rows based on number of columns

    # Set up a figure with the calculated number of rows and columns
    fig, axs = plt.subplots(rows, cols, figsize=(16, 4 * rows))  # Adjust height based on rows

    # Flatten axs for easy iteration if it's 2D
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    # Initialize lists to accumulate gamma and zeta values across brands
    all_gamma_best = []
    all_zeta_best = []
    all_labels = []
    all_brands = []

    for i, brand in enumerate(config["dataset"]["brand_list"]):
        print("Processing brand:", brand)
        config["dataset"]["brand"] = brand

        # Create unique folder names for each brand
        folder_name = f"""dataset/generated_results/train_results/multiple_brands/epoch{config["modelTraining"]["epoch"]}_run{config["simulationRecovery"]["independentRun"]}_{config["dataset"]["brand"]}"""
        os.makedirs(folder_name, exist_ok=True)

        # Update paths in configuration
        config["modelTraining"]["lossPath"] = f"{folder_name}/loss_results.csv"
        config["modelTraining"]["parametersPath"] = f"{folder_name}/parameters_results.csv"
        config["simulationRecovery"]["paramsSavedPath"] = f"{folder_name}/simulated_parameters.csv"
        config["simulationRecovery"]["simulatedParamSavedPath"] = f"{folder_name}/temp_simulated.csv"

        # Initialize DataPreprocessing with the updated brand
        data_preprocessing = DataPreprocessing(
            config["dataset"]["path"],
            brand,
            config["dataset"]["dependent_variable"],
            config["dataset"]["independent_variables_X"],
            config["dataset"]["independent_variables_Z"],
        )

        # Preprocess the data
        X_t, Z_t, Y_t = data_preprocessing.preprocess(normalization=True)

        # Run the simulation recovery on each subplot
        gamma_best, zeta_best, labels = trainer(X_t, Z_t, Y_t, plot_result=True, ax=axs[i])

        # Accumulate gamma and zeta values across brands for combined plot
        all_gamma_best.extend(gamma_best)
        all_zeta_best.extend(zeta_best)
        all_labels.extend(labels)
        all_brands.extend([brand] * len(gamma_best))  # Add brand names for each point

        # Add a title for each subplot
        axs[i].set_title(f"{brand} - Actual vs Predicted")

    # Hide any unused subplots if the grid is larger than the number of brands
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])  # Remove unused subplots

    # Plot all gamma and zeta best values on a single plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_gamma_zeta_combined(all_gamma_best, all_zeta_best, all_labels, all_brands, ax)
    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":

#     # Loop through each brand in brand_list
#     for brand in config["dataset"]["brand_list"]:
#         print("brand main", brand)
#         # Reload the config module to reset it to its original state
        
#         config["dataset"]["brand"] = brand


#         folder_name = f"""dataset/generated_results/train_results/multiple_brands/epoch{config["modelTraining"]["epoch"]}_run{config["simulationRecovery"]["independentRun"]}_{config["dataset"]["brand"]}"""
#         os.makedirs(folder_name, exist_ok=True)

#         # Update paths in configuration
#         config["modelTraining"]["lossPath"] = f"{folder_name}/loss_results.csv"
#         config["modelTraining"]["parametersPath"] = f"{folder_name}/parameters_results.csv"
#         config["simulationRecovery"]["paramsSavedPath"] = f"{folder_name}/simulated_parameters.csv"
#         config["simulationRecovery"]["simulatedParamSavedPath"] = f"{folder_name}/temp_simulated.csv"

#         # Initialize DataPreprocessing with the updated brand
#         data_preprocessing = DataPreprocessing(
#             config["dataset"]["path"],
#             brand,
#             config["dataset"]["dependent_variable"],
#             config["dataset"]["independent_variables_X"],
#             config["dataset"]["independent_variables_Z"],
#         )

#         # Preprocess the data
#         X_t, Z_t, Y_t = data_preprocessing.preprocess(normalization=True)

#         # Run the simulation recovery experiments
#         simulation_recovery(X_t, Z_t, Y_t)

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":

#     data_preprocessing = DataPreprocessing(
#         config["dataset"]["path"],
#         config["dataset"]["brand"],
#         config["dataset"]["dependent_variable"],
#         config["dataset"]["independent_variables_X"],
#         config["dataset"]["independent_variables_Z"],
#     )

#     X_t, Z_t, Y_t = data_preprocessing.preprocess(normalization=True)

#     """
#     Experiment 1
#     run the simulation recovery with independent run based on the 
#     config["simulationRecovery"]["independentRun"] to see the plot
#     """
#     trainer(X_t, Z_t, Y_t)

#     """
#     Experiment 2
#     it's a parallelly run of the above #1 simulation recovery with a list of independent runs, based on the 
#     config["simulationRecovery"]["independentRunList"] to see the parameters table.
#     """
#     # trainer_with_multi_independent_runs(X_t, Z_t, Y_t)

#     """
#     Experiment 3
#     it's a parallelly run of the above #1 simulation recovery with a list of independent runs, based on the 
#     config["simulationRecovery"]["independentRunList"] to see the parameters table.
#     """
