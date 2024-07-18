import pytest
import pandas as pd
import numpy as np
import sys
import os
import torch.nn as nn
import torch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from dynamic_linear_model.model import DynamicLinearModel
from dynamic_linear_model.model_origin import DynamicLinearModel as DynamicLinearModel_origin


@pytest.fixture
def mock_data():
    X_t = torch.tensor([[-0.33491841,  0.7392458 , -0.57032941, -0.43038056, -0.2558916 ,
                         -0.1680476 , -0.11805218],
                        [ 0.11755216, -0.37441825, -0.57032941, -0.46554489, -0.2558916 ,
                         -0.1680476 , -0.11805218]])
    
    Z_t = torch.tensor([[-0.51720119, -0.10946812], [-0.19904563,  0.42544762]])

    return X_t, Z_t

def test_dlm_model(mock_data):
    X_t, Z_t = mock_data
    dlm_model = DynamicLinearModel()
    dlm_model.eta = nn.Parameter(torch.tensor([ 0.5652,  0.4609, -0.0409,  0.0895,  0.0421,  0.0573,  0.0152]))
    dlm_model.zeta = nn.Parameter(torch.tensor([ 0.1010, -0.0955]))
    dlm_model.G = nn.Parameter(torch.tensor(0.7))
    predicted_Y = dlm_model(X_t, Z_t)
    assert [round(f, 4) for f in predicted_Y.data.tolist()] == [0.0931, -0.1979], "predicted_Y is not correct"

def test_dlm_model_origin(mock_data):
    X_t, Z_t = mock_data
    dlm_model = DynamicLinearModel_origin()
    dlm_model.eta = np.array([0.5652, 0.4609, -0.0409, 0.0895, 0.0421, 0.0573, 0.0152])
    dlm_model.zeta = np.array([ 0.1010, -0.0955])
    dlm_model.G = np.array(0.7)
    predicted_Y = dlm_model.dlm_model(X_t.numpy(), Z_t.numpy())  # Ensure you are passing numpy arrays if the original model expects it
    assert [round(f, 4) for f in predicted_Y.tolist()] == [0.0931, -0.1979], "predicted_Y is not correct"

if __name__ == "__main__":
    pytest.main()