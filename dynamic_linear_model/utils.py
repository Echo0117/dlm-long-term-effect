import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import config

def convert_normalized_data(scaler, data):
    convert_data = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    return convert_data

def plot(data_1, data_2, label1, label2, title, ax):
    # plt.figure(figsize=(10, 5))
    ax.plot(data_1, 'b-', label=label1)
    ax.plot(data_2, 'r--', label=label2)
    # Prepare the specific configuration text
    original_g = config["modelTraining"]["originalG"]
    config_text = ""
    if config["inferenceMethod"] == "mcmc":
        mcmc_params = config["inferenceParams"]["mcmc"]
        config_text += "inferenceParams: "
        for key, value in mcmc_params.items():
            config_text += f"  {key}: {value}\n"
    # if config["inferenceMethod"] == "torch_autograd":
    #     gd_params = config["modelTraining"]
    #     config_text += "inferenceParams: "
    #     for key, value in gd_params.items():
    #         if key != "modelPath" and key != "nSplits":
    #             config_text += f"  {key}: {value}\n"

    config_text += f"originalG: {original_g}"
    # ax.text(0.15, 0.8, config_text, fontsize=7, va='top', wrap=True)
    ax.set_title(
        f'Simulated G = {"{:.1f}".format(config["modelTraining"]["originalG"])}'
        )

def training_plot_metrics(train_losses, val_losses, test_loss, train_mse_list, val_mse_list, test_mse):
    """
    Plot training and validation loss and accuracy, and test metrics.

    Parameters:
    train_losses (list): List of training losses.
    val_losses (list): List of validation losses.
    test_loss (float): Test loss.
    train_mse_list (list): List of training accuracies.
    val_mse_list (list): List of validation accuracies.
    test_mse (float): Test mse.
    """
    epochs = range(1, config["modelTraining"]["epoch"] + 1)

    plt.figure(figsize=(14, 6))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    # plt.axhline(y=test_loss, color='g', linestyle='-', label='Test loss')
    plt.title('Training, Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mse_list, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_mse_list, 'ro-', label='Validation accuracy')
    # plt.axhline(y=test_mse, color='g', linestyle='-', label='Test accuracy')
    plt.title('Training, Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def negative_log_likelihood(params, Y_t, X_t, Z_t):
    """
    Calculate the negative log likelihood for the given parameters and data.

    Parameters:
    params (list): List of parameters (G, eta, zeta).
    Y_t (np.ndarray): Dependent variable vector Y.
    X_t (np.ndarray): Independent variables matrix X.
    Z_t (np.ndarray): Independent variables matrix Z.

    Returns:
    float: Negative log likelihood value.
    """
    T = len(Y_t)

    if config['inferenceMethod'] == "pertub_gradient_descent" or config['inferenceMethod'] == "mcmc":
        G, *coeffs = params
        # if config["modelTraining"]["addSigmoid"]:
        #     G = sigmoid(G)

        eta = np.array(coeffs[:X_t.shape[1]])
        zeta = np.array(coeffs[X_t.shape[1]:])
        theta_t = np.zeros(T)
        neg_log_likelihood = 0

        for t in range(T):
            if t > 0:
                theta_t[t] = G * theta_t[t-1] + np.dot(Z_t[t-1], zeta / 2)
            
            predicted_Y_t = theta_t[t] + np.dot(X_t[t], eta) + np.dot(Z_t[t], zeta / 2)
            neg_log_likelihood -= 0.5 * ((Y_t[t] - predicted_Y_t)**2)
    
    elif config['inferenceMethod'] == "torch_autograd":
        G, eta, zeta = params
        if config["modelTraining"]["addSigmoid"]:
            G = torch.sigmoid(G)

        theta_t = torch.zeros(T, dtype=torch.float32)
        neg_log_likelihood = torch.tensor(0.0, dtype=torch.float32)

        for t in range(T):
            if t > 0:
                theta_t[t] = G * theta_t[t-1].clone() + torch.dot(Z_t[t-1], zeta / 2)

            predicted_Y_t = theta_t[t] + torch.dot(X_t[t], eta) + torch.dot(Z_t[t], zeta / 2)
            neg_log_likelihood += 0.5 * ((Y_t[t] - predicted_Y_t)**2)

            # Add checks for numerical stability
            if torch.isnan(predicted_Y_t) or torch.isinf(predicted_Y_t):
                logging.warn(f"Numerical instability at step {t}. Predicted Y_t: {predicted_Y_t.item()}")
                break
    return neg_log_likelihood

def log_parameters_results(simulated_G=None, simulated_eta=None, simulated_zeta=None, params=None):
    """
    Log the simulation results to a log file.

    Parameters:
    simulated_G (float): The G parameter from the simulation.
    simulated_eta (np.ndarray): The eta parameter from the simulation.
    simulated_zeta (np.ndarray): The zeta parameter from the simulation.
    """
    logging.info("Simulation Results:")
    if params:
        logging.info(f"params: {params}")
    else:
        logging.info(f"Simulated G: {simulated_G}")
        logging.info(f"Simulated eta: {simulated_eta}")
        logging.info(f"Simulated zeta: {simulated_zeta}")