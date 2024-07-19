import pytest
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import sys
import pandas as pd
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from dynamic_linear_model.inference.gradient_descent import GradientDescentPerturbation


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
    
    return X_t, Z_t, Y_t, Y_t_normalized, scaler_Y

def test_gradient_descent_perturb(mock_data):
    X_t, Z_t, Y_t, Y_t_normalized, scaler_Y = mock_data

    # Initial parameters with noise
    original_eta = np.zeros(X_t.shape[1])
    original_zeta = np.zeros(Z_t.shape[1])
    initial_G = 0.7 + np.random.normal(0, 0.01)
    initial_eta = original_eta + np.random.normal(0, 0.01, size=original_eta.shape)
    initial_zeta = original_zeta + np.random.normal(0, 0.01, size=original_zeta.shape)

    initial_params = [initial_G] + list(initial_eta) + list(initial_zeta)

    # Initialize and optimize
    gd_perturb = GradientDescentPerturbation(learning_rate=0.001, n_iterations=500)
    optimized_params = gd_perturb.optimize(Y_t, X_t, Z_t, initial_params)

    G, *coeffs = optimized_params
    eta = np.array(coeffs[:X_t.shape[1]])
    zeta = np.array(coeffs[X_t.shape[1]:])

    print("Perturb Optimized G:", G)
    print("Perturb Optimized eta:", eta)
    print("Perturb Optimized zeta:", zeta)

    assert G is not None, "Optimized G is None"
    assert eta is not None, "Optimized eta is None"
    assert zeta is not None, "Optimized zeta is None"

if __name__ == "__main__":
    pytest.main()