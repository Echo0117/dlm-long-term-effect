import logging
import numpy as np
import torch
from config import config
from dynamic_linear_model.data_processing import DataPreprocessing
from dynamic_linear_model.data_simulation import DataSimulation, SimulationRecovery
from dynamic_linear_model.utils import convert_normalized_data, plot
from dynamic_linear_model.inference.prediction import SalesPrediction
from dynamic_linear_model.train import Trainer
import matplotlib.pyplot as plt
import os

# Configure logging
logging.basicConfig(filename='log.txt', filemode='w',level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main(X_t, Z_t, Y_t, do_data_simulation=False):
    """
    Main function to run the data preprocessing, training, and plotting.

    Parameters:
    X_t (np.ndarray): Independent variables matrix X.
    Z_t (np.ndarray): Independent variables matrix Z.
    Y_t (np.ndarray): Dependent variable vector Y.
    do_data_simulation (bool): Flag to indicate if data simulation should be performed.
    """
    # Specify the path to the CSV file
    file_path = config["simulationRecovery"]["paramsSavedPath"]
    file_simulated_path = config["simulationRecovery"]["simulatedParamSavedPath"]
    # Check if the file exists before attempting to delete it
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} has been deleted successfully.")
    elif os.path.exists(file_simulated_path):
        os.remove(file_simulated_path)
        print(f"File {file_simulated_path} has been deleted successfully.")
    else:
        print(f"File {file_path} and {file_simulated_path} does not exist.")

    # G = config["modelTraining"]["originalG"]
    G_list = np.arange(0.1, 1.0, 0.2)
    # G_list = [0.1, 0.6, 0.9]
    fig, axs = plt.subplots(len(G_list), 2, figsize=(16, 10))
    fig_optim_g, axs_optim_g = plt.subplots(len(G_list), config["simulationRecovery"]["independentRun"], figsize=(18, 12))
    fig_training , axs_training = plt.subplots(len(G_list), config["simulationRecovery"]["independentRun"], figsize=(18, 12))
    for G, ax, ax_training, ax_optim_g in zip(G_list, axs, axs_training, axs_optim_g):
        config["modelTraining"]["originalG"] = G
        # Convert to torch tensors if using torch_autograd method
        if config['inferenceMethod'] == "torch_autograd":
            X_t = torch.tensor(X_t, dtype=torch.float32)
            Z_t = torch.tensor(Z_t, dtype=torch.float32)

        # Data simulation
        if do_data_simulation:
            data_simulation = DataSimulation(X_t, Z_t, Y_t, G, config["modelTraining"]["theta0"])
            simulated_results = data_simulation.get_simulation_results()
            Y_t = simulated_results['simulated_Y']
            _, _, scaler_Y = data_preprocessing.get_normalized_scaler()
            # converted_Y_t = convert_normalized_data(scaler_Y, Y_t)
            # print("Y_tY_tY_t", Y_t)
            # plt.plot(Y_t, label=f'indenpendt run when G = {G}')
        
            # # plot(converted_Y_t, converted_Y_t, 'Simulated Y_t', 'Predicted Y_t', "Simulated vs Predicted", ax[1])
            Y_predicted_list = SimulationRecovery(X_t, Z_t, Y_t).recovery_for_simulation(ax[0], ax_training, ax_optim_g)
            predicted_Y = np.mean(Y_predicted_list, axis=0)
            # return True

            # # Train the model
            # trainer = Trainer(X_t, Z_t, Y_t)
            # trainer.train()

            # Predict using the trained model
            # sales_prediction = SalesPrediction()
            # predicted_Y = sales_prediction.predict(X_t, Z_t)

            # Convert normalized data back to original scale
            # _, _, scaler_Y = data_preprocessing.get_normalized_scaler()
            # converted_Y_t = convert_normalized_data(scaler_Y, Y_t)
            # converted_predicted_Y = convert_normalized_data(scaler_Y, predicted_Y)

            converted_Y_t = Y_t
            converted_predicted_Y = predicted_Y

            # Plot actual vs simulated if data simulation was performed
            if do_data_simulation:
                plot(converted_Y_t, converted_predicted_Y, 'Simulated Y_t', 'Predicted Y_t', "Simulated vs Predicted", ax[1])
                continue
            else:
                plot(converted_Y_t, converted_predicted_Y, 'Actual Y_t', 'Predicted Y_t', "Actual vs Predicted", ax[1])
    
    fig_optim_g.supxlabel('Epoch')
    fig_optim_g.supylabel('value')
    fig_optim_g.suptitle('change of g value during optimizing')
    handles, labels = [], []
    
    for handle in fig_optim_g.axes[0].get_lines():
        handles.append(handle)
        labels.append(handle.get_label())
    fig_optim_g.legend(handles, labels, loc='upper right', ncol=2)
    fig_optim_g.tight_layout()

    fig_training.suptitle(f"Change of Loss during Recovery")
    fig_training.supxlabel("Epoch")
    fig_training.supylabel("Loss")
    handles, labels = [], []
    for ax in fig_training.axes:
        for handle in ax.get_lines():
            handles.append(handle)
            labels.append(handle.get_label())
    fig_training.legend(handles[:1], labels, loc='upper right', ncol=2)
    fig_training.tight_layout()

    fig.axes[0].set_title(f"Change of parameters with different run")
    fig.axes[0].set_xlabel("Independent run")
    fig.axes[0].set_ylabel("Value")

    fig.axes[1].set_title(f"Simulated vs Predicted")
    fig.axes[1].set_xlabel("Time")
    fig.axes[1].set_ylabel("Volume Sold")

    handles, labels = [], []
    for handle in fig.axes[0].get_lines():
        handles.append(handle)
        labels.append(handle.get_label())
    fig.axes[0].legend(handles, labels, loc='upper right', ncol=2)

    handles, labels = [], []
    for handle in fig.axes[1].get_lines():
        handles.append(handle)
        labels.append(handle.get_label())
    fig.axes[1].legend(handles, labels, loc='upper left', ncol=2)

    config_text = ""
    if config["inferenceMethod"] == "torch_autograd":
        gd_params = config["modelTraining"]
        config_text += "inferenceParams: "
        for key, value in gd_params.items():
            if key != "modelPath" and key != "nSplits":
                config_text += f"  {key}: {value}\n"
    fig.text(0.05, 0.1, config_text, fontsize=7, va='top', wrap=True)
    fig.tight_layout()

    plt.tight_layout()
    plt.show()

    # # Add labels and legend
    # plt.xlabel('X-axis label')  # Replace with your specific x-axis label
    # plt.ylabel('Y-axis label')  # Replace with your specific y-axis label
    # plt.title('Simulation Results for Different G values')  # Replace with your specific title
    # plt.legend()  # Show the legend

    # # Display the plot
    # plt.show()

if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(config["dataset"]["path"], 
                    config["dataset"]["brand"], 
                    config["dataset"]["dependent_variable"], 
                    config["dataset"]["independent_variables_X"], 
                    config["dataset"]["independent_variables_Z"])

    X_t, Z_t, Y_t = data_preprocessing.preprocess(normalization=True)
    main(X_t, Z_t, Y_t, do_data_simulation=True)

    # TODO: 
    # initialization of parameters and optimization of parameters in gd
    # quantify the long term effect of the marketing campaign on sales
    # param recovery
 


 # lr e-4
 # g: sigmoid
 # when g=0.1 or g=0.9, (simulation part) when does the problem occur, g is big or small
 # how gradient steps effect the g, plot how g changes
 # mcmc (if above didn't work, try mcmc)

# check if monthly meeting is necessary
# create notion phd journal