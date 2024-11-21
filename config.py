import json5
import os

import pandas as pd


root_path = os.path.dirname(os.path.abspath(__file__))
"""
read json config file。
"""
f = open(root_path + "/config.json", encoding="utf-8")

config = json5.load(f)

config["dataset"]["xDim"] = len(config["dataset"]["independent_variables_X"])
config["dataset"]["zDim"] = len(config["dataset"]["independent_variables_Z"])
config["dataset"]["path"] = os.path.join(root_path, config["dataset"]["path"])
config["simulationRecovery"]["paramsSavedPath"] = os.path.join(
    root_path, config["simulationRecovery"]["paramsSavedPath"]
)
config["simulationRecovery"]["simulatedParamSavedPath"] = os.path.join(
    root_path, config["simulationRecovery"]["simulatedParamSavedPath"]
)

# folder_name = f"""dataset/generated_results/train_results/multiple_brands/epoch{config["modelTraining"]["epoch"]}_run{config["simulationRecovery"]["independentRun"]}_{config["dataset"]["brand"]}"""
# os.makedirs(folder_name, exist_ok=True)

# # Update paths in configuration
# config["modelTraining"]["lossPath"] = f"{folder_name}/loss_results.csv"
# config["modelTraining"]["parametersPath"] = f"{folder_name}/parameters_results.csv"
# config["simulationRecovery"]["paramsSavedPath"] = f"{folder_name}/simulated_parameters.csv"
# config["simulationRecovery"]["simulatedParamSavedPath"] = f"{folder_name}/temp_simulated.csv"


# import torch
# config["device"] = torch.device("mps" if torch.cuda.is_available() else "cpu")