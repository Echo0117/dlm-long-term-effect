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

# Configure logging
logging.basicConfig(filename='log.txt', filemode='w',level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

import os



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
    # G_list = np.arange(0, 1.1, 0.2)
    G_list = [0.4, 0.6, 0.8]
    fig, axs = plt.subplots(len(G_list), 2, figsize=(18, 12))
    # axs = axs.flatten()
    for G, ax in zip(G_list, axs):
        print(ax)
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
            Y_predicted_list = SimulationRecovery(X_t, Z_t, Y_t).recovery_for_simulation(ax[0])
            predicted_Y = Y_predicted_list[0]
            # return True

            # # Train the model
            # trainer = Trainer(X_t, Z_t, Y_t)
            # trainer.train()

            # Predict using the trained model
            # sales_prediction = SalesPrediction()
            # predicted_Y = sales_prediction.predict(X_t, Z_t)

            # Convert normalized data back to original scale
            _, _, scaler_Y = data_preprocessing.get_normalized_scaler()
            converted_Y_t = convert_normalized_data(scaler_Y, Y_t)
            converted_predicted_Y = convert_normalized_data(scaler_Y, predicted_Y)

            # Plot actual vs simulated if data simulation was performed
            if do_data_simulation:
                # plot(simulated_results['actual_Y'], simulated_results['simulated_Y'], 'Actual Y_t', 'Simulated Y_t', "Simulated vs Actual")
                plot(converted_Y_t, converted_predicted_Y, 'Simulated Y_t', 'Predicted Y_t', "Simulated vs Predicted", ax[1])
            else:
                plot(converted_Y_t, converted_predicted_Y, 'Actual Y_t', 'Predicted Y_t', "Actual vs Predicted", ax[1])

    plt.tight_layout()
    plt.show()


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