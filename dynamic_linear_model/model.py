import time
import pickle
import random
import torch
import torch.nn as nn
from config import config

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.benchmark = True

device = config["device"]


class DynamicLinearModel(nn.Module):
    def __init__(self, num_runs, model_path=None):
        super(DynamicLinearModel, self).__init__()
        self.num_runs = num_runs
        
        # Initialize parameters for all runs
        # self.G = nn.Parameter(
        #     torch.tensor([random.uniform(-4, 4) for _ in range(num_runs)], device=device)
        # )  # State transition coefficients
        # self.G = nn.Parameter(
        #     torch.FloatTensor(num_runs).uniform_(-4, 4).to(device)
        # ) # State transition coefficients
        self.G = nn.Parameter(
            torch.full((num_runs,), 4.595, device=device, requires_grad=False)) # sigmoid(2.1972) = 0.9; sigmoid(4.595) = 0.99; sigmoid(3.89) = 0.98; sigmoid(0.4055) = 0.6; 1.3863=0.8
        self.G.requires_grad = False # we need to set up this, otherwise requires_grad will still be true
        
        self.eta = nn.Parameter(
            torch.abs(torch.randn(num_runs, config["dataset"]["xDim"], device=device))
        )  # Coefficients for X_t
        self.zeta = nn.Parameter(
            torch.randn(num_runs, config["dataset"]["zDim"], device=device)
        )  # Coefficients for Z_t
        self.gamma = nn.Parameter(
            torch.randn(num_runs, config["dataset"]["zDim"], device=device)
        )  # Coefficients for Z_t-1
        self.sigmoid = nn.Sigmoid()


    def forward(self, X_t: torch.Tensor, Z_t: torch.Tensor, is_simulation: bool = False) -> tuple:

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

        # Initialize predicted_Y for all runs
        predicted_Y = torch.zeros(self.num_runs, T, device=device)

        # Prepare G_hat based on is_add_sigmoid
        if is_add_sigmoid:
            G_hat = self.sigmoid(self.G)
        else:
            G_hat = self.G

        # Repeat X_t and Z_t for all runs
        # X_t_repeated = X_t.unsqueeze(0).repeat(self.num_runs, 1, 1)
        # Z_t_repeated = Z_t.unsqueeze(0).repeat(self.num_runs, 1, 1)

        X_t_repeated = X_t.unsqueeze(0).expand(self.num_runs, -1, -1)
        Z_t_repeated = Z_t.unsqueeze(0).expand(self.num_runs, -1, -1)


        # Initialize theta for all runs
        theta = torch.zeros(self.num_runs, device=device)

        for t in range(T):
            if t > 0:
                theta = G_hat * theta + torch.sum(Z_t_repeated[:, t - 1] * gamma_hat, dim=1)
            predicted_Y[:, t] = (
                theta + torch.sum(X_t_repeated[:, t] * eta_hat, dim=1) + torch.sum(Z_t_repeated[:, t] * zeta_hat, dim=1)
            )

        return predicted_Y

    
    def save_model(self, model_path: str) -> None:
        model_data = {
            "G": self.G.detach().cpu().numpy(),
            "eta": self.eta.detach().cpu().numpy(),
            "zeta": self.zeta.detach().cpu().numpy(),
            "gamma": self.gamma.detach().cpu().numpy(),
        }
        with open(model_path, "wb") as file:
            pickle.dump(model_data, file)

    # def _load_model(self, model_path: str) -> None:
    #     with open(model_path, "rb") as file:
    #         model_data = pickle.load(file)
    #         self.G = nn.Parameter(torch.tensor(model_data["G"], device=device))
    #         self.eta = nn.Parameter(torch.tensor(model_data["eta"], device=device))
    #         self.zeta = nn.Parameter(torch.tensor(model_data["zeta"], device=device))
    #         self.gamma = nn.Parameter(torch.tensor(model_data["gamma"], device=device))
