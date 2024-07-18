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
        self.G = nn.Parameter(torch.tensor(random.uniform(0, 1)))  # State transition coefficient
        self.eta = nn.Parameter(torch.randn(config["dataset"]["xDim"]))  # Coefficients for X_t
        self.zeta = nn.Parameter(torch.randn(config["dataset"]["zDim"]))  # Coefficients for Z_t

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

    def forward(self, X_t, Z_t):
        """
        Apply the Dynamic Linear Model (DLM) to predict values.

        Parameters:
        X_t (torch.Tensor): Independent variables matrix X.
        Z_t (torch.Tensor): Independent variables matrix Z.

        Returns:
        torch.Tensor: Predicted values.
        """
        T = X_t.size(0)
        theta_t = torch.zeros(T)
        predicted_Y = torch.zeros(T)
        # print("model self.G", self.G)
        # print("model self.eta", self.eta)
        # print("model self.zeta", self.zeta)
        for t in range(T):
            if t > 0:
                if config["modelTraining"]["addSigmoid"]:
                    G_hat = torch.sigmoid(self.G)
                else:
                    G_hat = self.G 
                theta_t[t] = G_hat * theta_t[t - 1].clone() + torch.dot(Z_t[t - 1], self.zeta / 2)
                # print("G_hat", G_hat)
                # print("G_hat", Z_t)

                # print("theta_ttheta_t", theta_t)

            predicted_Y[t] = theta_t[t] + torch.dot(X_t[t], self.eta) + torch.dot(Z_t[t], self.zeta / 2)
        return predicted_Y, self.G.clone().data, self.eta.clone().data, self.zeta.clone().data

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

# # Training function
# def train_model(model, X_t, Z_t, Y_t, num_epochs=100, learning_rate=0.01):
#     """
#     Train the Dynamic Linear Model.

#     Parameters:
#     model (DynamicLinearModel): The model to train.
#     X_t (torch.Tensor): Independent variables matrix X.
#     Z_t (torch.Tensor): Independent variables matrix Z.
#     Y_t (torch.Tensor): Dependent variable vector Y.
#     num_epochs (int): Number of epochs to train.
#     learning_rate (float): Learning rate for the optimizer.
#     """
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     for epoch in range(num_epochs):
#         model.train()

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = model(X_t, Z_t)
#         loss = criterion(outputs, Y_t)

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         if (epoch+1) % 10 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# # Example usage
# if __name__ == "__main__":
#     # Dummy data
#     X_t = torch.randn(100, 5)  # 100 time steps, 5 features
#     Z_t = torch.randn(100, 5)  # 100 time steps, 5 features
#     Y_t = torch.randn(100)     # 100 time steps

#     # Initialize model
#     model = DynamicLinearModel()

#     # Train model
#     train_model(model, X_t, Z_t, Y_t, num_epochs=100, learning_rate=0.01)

#     # Save the trained model
#     model.save_model("trained_dlm.pkl")
