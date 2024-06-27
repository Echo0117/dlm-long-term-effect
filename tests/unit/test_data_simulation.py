import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
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
    df = df[df["brand"] == "absolut"]

    # Define dependent and independent variables
    Y_t = df['volume_so_off'].values
    X_t = df[['relative_gap_to_90th_price_off_off', 'off_trade_visibility_off', 'digital_off', 'digital_social_media_off', 'out_of_home_off', 'television_off', 'brand_experience_off']].values
    Z_t = df[['distribution_off_off', 'discount_price_comp_to_pr_off_off']].values

    # Normalize the data
    scaler_X = StandardScaler()
    scaler_Z = StandardScaler()
    scaler_Y = StandardScaler()

    X_t = scaler_X.fit_transform(X_t)
    Z_t = scaler_Z.fit_transform(Z_t)
    Y_t_normalized = scaler_Y.fit_transform(Y_t.reshape(-1, 1)).flatten()
    
    return X_t, Z_t, Y_t, Y_t_normalized

def test_data_simulation(mock_data):
    X_t, Z_t, Y_t, Y_t_normalized = mock_data
    
    # Initialize DataSimulation
    data_simulation = DataSimulation(X_t, Z_t, Y_t_normalized)
    
    # Test obtain_parameters_by_lr
    data_simulation.obtain_parameters_by_lr()
    assert data_simulation.eta is not None, "eta should not be None"
    assert data_simulation.zeta is not None, "zeta should not be None"

    # Test generate_theta
    theta_t = data_simulation.generate_theta()
    assert theta_t.shape == (len(Y_t_normalized),), "theta_t shape mismatch"
    
    # Test generate_simulated_Y
    predicted_Y = data_simulation.generate_simulated_Y()
    assert predicted_Y.shape == (len(Y_t_normalized),), "predicted_Y shape mismatch"
    
    # Test get_results_df
    results_df = data_simulation.get_results_df()
    assert 'Actual_Y' in results_df.columns, "Results DataFrame should contain 'Actual_Y'"
    assert 'Predicted_Y' in results_df.columns, "Results DataFrame should contain 'Predicted_Y'"
    assert results_df.shape[0] == len(Y_t), "Results DataFrame row count mismatch"
    
    # Check if the normalization has zero mean and unit variance
    assert np.allclose(predicted_Y.mean(), -0.018889871620384285, atol=1e-7), "predicted_Y mean is not correct"
    assert np.allclose(predicted_Y.std(), 1.070456049238724, atol=1e-7), "predicted_Y std is not correct"

if __name__ == "__main__":
    pytest.main()