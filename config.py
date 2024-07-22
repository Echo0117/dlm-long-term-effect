import json5
import os


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

# Check if MPS is available and set the device
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
config["device"] = device
