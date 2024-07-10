import json
import os


root_path = os.path.dirname(os.path.abspath(__file__))

"""
read json config file。
"""
f = open(root_path + "/config.json", encoding='utf-8')

config = json.load(f)