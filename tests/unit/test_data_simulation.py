import pytest
import numpy as np
import pandas as pd
import sys
import os
import torch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from config import config
from dynamic_linear_model.data_processing import DataPreprocessing
from dynamic_linear_model.data_simulation import DataSimulation


@pytest.fixture
def mock_data():
    data = {
        'brand': ['absolut', 'absolut', 'absolut', 'absolut', 'absolut', 'absolut'],
        'volume_so_off': [48684.79, 101398.01, 49442.96, 160616.96, 165683.84, 123858.63],
        'relative_gap_to_90th_price_off_off': [0.067416448, 0.164185595, 0.050264697, 0.217222795, 0.188346898, 0.156085734],
        'discount_price_comp_to_pr_off_off': [-0.384771404, -0.298375116, -0.319037948, -0.169343185, -0.390948848, -0.358297856],
        'distribution_off_off': [0.285544535, 0.336966316, 0.319825723, 0.302685129, 0.35410691, 0.302685129],
        'off_trade_visibility_off': [5614.708487, 22458.83395, 11229.41697, 36495.60517, 25266.18819, 25266.18819],
        'digital_off': [0, 0, 0, 0, 0, 0],
        'digital_social_media_off': [0, 31419.98, 43892.66, 52123.99, 43481.99, 48931.33],
        'out_of_home_off': [0, 19470.39, 25672.4, 17106.11, 30878.62, 33065.08],
        'television_off': [385833.33, 360500, 425000, 425000, 443333.33, 207333.33],
        'brand_experience_off': [3525.26, 756.03, 0, 0, 0, 16375.53]
    }
    df = pd.DataFrame(data)
    df.to_csv('test_data.csv', index=False)
    return 'test_data.csv'

def test_data_simulation_mockdata(mock_data):
    file_path = mock_data
    brand = "absolut"
    dependent_variable = 'volume_so_off'
    independent_variables_X = [
        'relative_gap_to_90th_price_off_off', 'off_trade_visibility_off', 'digital_off',
        'digital_social_media_off', 'out_of_home_off', 'television_off', 'brand_experience_off'
    ]
    independent_variables_Z = [
        'distribution_off_off', 'discount_price_comp_to_pr_off_off'
    ]

    df = pd.read_csv(file_path)
    df = df[df["brand"] == brand]

    # Define dependent and independent variables
    Y_t = df[dependent_variable].values
    X_t = df[independent_variables_X].values
    Z_t = df[independent_variables_Z].values

    data_processor = DataPreprocessing(file_path, brand, dependent_variable, independent_variables_X, independent_variables_Z)
    X_t, Z_t, Y_t = data_processor.preprocess(normalization=True)

    X_t = torch.tensor(X_t, dtype=torch.float32)
    Z_t = torch.tensor(Z_t, dtype=torch.float32)
    Y_t = torch.tensor(Y_t, dtype=torch.float32)

    # Initialize DataSimulation
    data_simulation = DataSimulation(X_t, Z_t, Y_t, G=0.7)
    # Test obtain_parameters_by_lr
    data_simulation._obtain_initial_parameters_by_lr()
    assert [round(f, 4) for f in data_simulation.eta.data.tolist()] == [0.668, 0.4975, 0.0, 0.1225, 0.0648, 0.4217, 0.1615], "eta is not correct"
    assert [round(f, 4) for f in data_simulation.zeta.data.tolist()] == [-0.1799, -0.4271], "zeta is not correct"

    # Test generate_simulated_Y
    simulated_Y = data_simulation.generate_simulated_Y()
    assert simulated_Y.shape == (len(Y_t),), "Simulated_Y shape mismatch"

    # Check if the normalization has zero mean and unit variance
    assert np.allclose(simulated_Y.detach().numpy().mean(), -0.0188898, atol=1e-7), "Simulated_Y mean is not correct"
    assert np.allclose(simulated_Y.detach().numpy().std(), 1.0704558, atol=1e-7), "Simulated_Y std is not correct"

def test_data_simulation_df_germany():
    file_path = config["dataset"]["path"]
    brand = "absolut"
    dependent_variable = 'volume_so_off'
    independent_variables_X = [
        'relative_gap_to_90th_price_off_off', 'off_trade_visibility_off', 'digital_off',
        'digital_social_media_off', 'out_of_home_off', 'television_off', 'brand_experience_off'
    ]
    independent_variables_Z = [
        'distribution_off_off', 'discount_price_comp_to_pr_off_off'
    ]

    df = pd.read_csv(file_path)
    df = df[df["brand"] == brand]

    # Define dependent and independent variables
    Y_t = df[dependent_variable].values
    X_t = df[independent_variables_X].values
    Z_t = df[independent_variables_Z].values

    data_processor = DataPreprocessing(file_path, brand, dependent_variable, independent_variables_X, independent_variables_Z)
    X_t, Z_t, Y_t = data_processor.preprocess(normalization=True)

    X_t = torch.tensor(X_t, dtype=torch.float32)
    Z_t = torch.tensor(Z_t, dtype=torch.float32)
    Y_t = torch.tensor(Y_t, dtype=torch.float32)

    # Initialize DataSimulation
    data_simulation = DataSimulation(X_t, Z_t, Y_t, G=0.7)

    # Test obtain_parameters_by_lr
    data_simulation._obtain_initial_parameters_by_lr()
    assert [round(f, 4) for f in data_simulation.eta.data.tolist()] == [0.5652, 0.4609, -0.0409, 0.0895, 0.0421, 0.0573, 0.0152], "eta is not correct"
    assert [round(f, 4) for f in data_simulation.zeta.data.tolist()] == [0.101, -0.0955], "zeta is not correct"

    # Test generate_simulated_Y
    simulated_Y = data_simulation.generate_simulated_Y()
    assert simulated_Y.shape == (len(Y_t),), "Simulated_Y shape mismatch"

    # Check if the normalization has zero mean and unit variance
    assert np.allclose(simulated_Y.detach().numpy().mean(), 0.0092, atol=1e-4), "Simulated_Y mean is not correct"
    assert np.allclose(simulated_Y.detach().numpy().std(), 0.9254, atol=1e-4), "Simulated_Y std is not correct"

if __name__ == "__main__":
    pytest.main()