import torch
import torch.nn as nn
import pickle
from config import config
import random

torch.autograd.set_detect_anomaly(True)


class DynamicLinearModel(nn.Module):
    def __init__(self, model_path=None):
        """
        Initialize the Dynamic Linear Model class.

        Parameters:
        model_path (str): Path to the saved model parameters.
        """
        super(DynamicLinearModel, self).__init__()
        self.G = nn.Parameter(
            torch.tensor(random.uniform(0, 1))
        )  # State transition coefficient
        self.eta = nn.Parameter(
            torch.randn(config["dataset"]["xDim"])
        )  # Coefficients for X_t
        self.zeta = nn.Parameter(
            torch.randn(config["dataset"]["zDim"])
        )  # Coefficients for Z_t
        self.gamma = nn.Parameter(
            torch.randn(config["dataset"]["zDim"])
        )
        self.sigmoid = nn.Sigmoid()

        # if model_path and os.path.exists(model_path):
        #     self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Load the model parameters from a file.

        Parameters:
        model_path (str): Path to the saved model parameters.
        """
        with open(model_path, "rb") as file:
            model_data = pickle.load(file)
            self.G = nn.Parameter(torch.tensor(model_data["G"]))
            self.eta = nn.Parameter(torch.tensor(model_data["eta"]))
            self.zeta = nn.Parameter(torch.tensor(model_data["zeta"]))

    def forward(self, X_t, Z_t, is_simulation=False):
        """
        Apply the Dynamic Linear Model (DLM) to predict values.

        Parameters:
        X_t (torch.Tensor): Independent variables matrix X.
        Z_t (torch.Tensor): Independent variables matrix Z.

        Returns:
        torch.Tensor: Predicted values.
        """
        if is_simulation:
            is_add_sigmoid = False
        else:
            is_add_sigmoid = config["modelTraining"]["addSigmoid"]

        T = X_t.size(0)
        theta = 0
        predicted_Y = torch.zeros(T)

        
        eta_hat = self.eta
        zeta_hat = self.zeta
        gamma_hat = self.gamma
        print("model self.G", self.G)
        print("model self.eta", self.eta)
        print("model self.zeta", self.zeta)
        if is_add_sigmoid:
            G_hat = self.sigmoid(self.G)
        else:
            G_hat = self.G
        for t in range(T):
            if t > 0:

                    # eta_hat = self.sigmoid(self.eta)
                    # zeta_hat = self.sigmoid(self.zeta)

                theta = G_hat * theta + torch.dot(Z_t[t - 1], gamma_hat)
                # print("G_hat", G_hat)
                # print("G_hat", Z_t)
                # print("theta_ttheta_t", theta_t)

            predicted_Y[t] = (
                theta + torch.dot(X_t[t], eta_hat) + torch.dot(Z_t[t], zeta_hat)
            )
        return (
            predicted_Y,
            self.G.clone().data,
            self.eta.clone().data,
            self.zeta.clone().data,
        )

    def save_model(self, model_path):
        """
        Save the model parameters to a file.

        Parameters:
        model_path (str): Path to save the model parameters.
        """
        model_data = {
            "G": self.G.clone().detach().numpy(),
            "eta": self.eta.clone().detach().numpy(),
            "zeta": self.zeta.clone().detach().numpy(),
        }
        with open(model_path, "wb") as file:
            pickle.dump(model_data, file)
