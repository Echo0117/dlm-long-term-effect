import torch
import torch.nn as nn
import pickle
import random
from config import config


# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

device = config["device"]


class DynamicLinearModel(nn.Module):
    def __init__(self, model_path=None):
        """
        Initialize the Dynamic Linear Model class.

        Parameters:
        model_path (str, optional): Path to the saved model parameters.
        """
        super(DynamicLinearModel, self).__init__()
        self.G = nn.Parameter(
            torch.tensor(random.uniform(0, 1), device=device)
        )  # State transition coefficient
        self.eta = nn.Parameter(
            torch.randn(config["dataset"]["xDim"], device=device)
        )  # Coefficients for X_t
        self.zeta = nn.Parameter(
            torch.randn(config["dataset"]["zDim"], device=device)
        )  # Coefficients for Z_t
        self.gamma = nn.Parameter(
            torch.randn(config["dataset"]["zDim"], device=device)
        )  # Coefficients for Z‘_t-1
        self.sigmoid = nn.Sigmoid()

        # if model_path and os.path.exists(model_path):
        #     self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """
        Load the model parameters from a file.

        Parameters:
        model_path (str): Path to the saved model parameters.
        """
        with open(model_path, "rb") as file:
            model_data = pickle.load(file)
            self.G = nn.Parameter(torch.tensor(model_data["G"], device=device))
            self.eta = nn.Parameter(torch.tensor(model_data["eta"], device=device))
            self.zeta = nn.Parameter(torch.tensor(model_data["zeta"], device=device))
            self.gamma = nn.Parameter(torch.tensor(model_data["gamma"], device=device))

    def forward(
        self, X_t: torch.Tensor, Z_t: torch.Tensor, is_simulation: bool = False
    ) -> tuple:
        """
        Apply the Dynamic Linear Model (DLM) to predict values.

        Parameters:
        X_t (torch.Tensor): Independent variables matrix X.
        Z_t (torch.Tensor): Independent variables matrix Z.
        is_simulation (bool, optional): Flag indicating if the model is in simulation mode.

        Returns:
        tuple: A tuple containing predicted values, G, eta, zeta, and gamma.
        """
        X_t = X_t.to(device)
        Z_t = Z_t.to(device)

        eta_hat = self.eta
        zeta_hat = self.zeta
        gamma_hat = self.gamma

        if is_simulation:
            is_add_sigmoid = False
        else:
            is_add_sigmoid = config["modelTraining"]["addSigmoid"]

        T = X_t.size(0)
        theta = torch.tensor(0, device=device)
        predicted_Y = torch.zeros(T, device=device)

        if is_add_sigmoid:
            G_hat = self.sigmoid(self.G)
        else:
            G_hat = self.G

        for t in range(T):
            if t > 0:
                theta = G_hat * theta + torch.dot(Z_t[t - 1], gamma_hat)
            predicted_Y[t] = (
                theta + torch.dot(X_t[t], eta_hat) + torch.dot(Z_t[t], zeta_hat)
            )

        return (
            predicted_Y,
            G_hat.item(),
            eta_hat.data.cpu().numpy(),
            zeta_hat.data.cpu().numpy(),
            gamma_hat.data.cpu().numpy(),
        )

    def save_model(self, model_path: str) -> None:
        """
        Save the model parameters to a file.

        Parameters:
        model_path (str): Path to save the model parameters.
        """
        model_data = {
            "G": self.G.detach().cpu().numpy(),
            "eta": self.eta.detach().cpu().numpy(),
            "zeta": self.zeta.detach().cpu().numpy(),
        }
        with open(model_path, "wb") as file:
            pickle.dump(model_data, file)
