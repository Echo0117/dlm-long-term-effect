import json
import os


root_path = os.path.dirname(os.path.abspath(__file__))

"""
read json config file。
"""
f = open(root_path + "/config.json", encoding="utf-8")

config = json.load(f)

config["dataset"]["xDim"] = len(config["dataset"]["independent_variables_X"])
config["dataset"]["zDim"] = len(config["dataset"]["independent_variables_Z"])
