import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from dynamic_linear_model.data_processing import DataPreprocessing


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

def test_data_preprocessing(mock_data):
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

    data_processor = DataPreprocessing(file_path, brand, dependent_variable, independent_variables_X, independent_variables_Z)
    X_t, Z_t, Y_t = data_processor.preprocess(normalization=True)

    print(X_t.shape)

    assert X_t.shape == (6, 7), "X_t shape mismatch"
    assert Z_t.shape == (6, 2), "Z_t shape mismatch"
    assert Y_t.shape == (6,), "Y_t shape mismatch"

    assert np.allclose(X_t.mean(axis=0), [-2.59052039e-16, -2.03540888e-16,  0.00000000e+00,  0.00000000e+00,
 -1.85037171e-16, -1.48029737e-16,  7.40148683e-17], atol=1e-7), "X_t mean is not correct"
    assert np.allclose(X_t.std(axis=0), [1., 1., 0., 1., 1., 1., 1.], atol=1e-7), "X_t std is not correct"
    assert np.allclose(Z_t.mean(axis=0), [-1.85037171e-17,  3.51570624e-16], atol=1e-7), "Z_t mean is not correct"
    assert np.allclose(Z_t.std(axis=0), [1., 1.], atol=1e-7), "Z_t std is not correct"
    assert np.allclose(Y_t.mean(), 1.3877787807814457e-16, atol=1e-7), "Y_t mean is not correct"
    assert np.allclose(Y_t.std(), 1.0, atol=1e-7), "Y_t std is not correct"

if __name__ == "__main__":
    pytest.main()