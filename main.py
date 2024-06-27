import logging
import torch
from config import config
from dynamic_linear_model.data_processing import DataPreprocessing
from dynamic_linear_model.data_simulation import DataSimulation
from dynamic_linear_model.utils import convert_normalized_data, plot
from dynamic_linear_model.inference.prediction import SalesPrediction
from dynamic_linear_model.train import Trainer
import matplotlib.pyplot as plt

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

    # Data simulation
    if do_data_simulation:
        data_simulation = DataSimulation(X_t, Z_t, Y_t, config["modelTraining"]["originalG"], config["modelTraining"]["theta0"])
        simulated_results = data_simulation.get_simulation_results()
        Y_t = simulated_results['simulated_Y']

    # Convert to torch tensors if using torch_autograd method
    if config['inferenceMethod'] == "torch_autograd":
        X_t = torch.tensor(X_t, dtype=torch.float32)
        Z_t = torch.tensor(Z_t, dtype=torch.float32)

    # Train the model
    trainer = Trainer(X_t, Z_t, Y_t)
    trainer.train()

    # Predict using the trained model
    sales_prediction = SalesPrediction()
    predicted_Y = sales_prediction.predict(X_t, Z_t)

    # Convert normalized data back to original scale
    _, _, scaler_Y = data_preprocessing.get_normalized_scaler()
    converted_Y_t = convert_normalized_data(scaler_Y, Y_t)
    converted_predicted_Y = convert_normalized_data(scaler_Y, predicted_Y)

    # Plot actual vs simulated if data simulation was performed
    if do_data_simulation:
        plot(simulated_results['actual_Y'], simulated_results['simulated_Y'], 'Actual Y_t', 'Simulated Y_t', "Simulated vs Actual")
        plot(converted_Y_t, converted_predicted_Y, 'Simulated Y_t', 'Predicted Y_t', "Simulated vs Predicted")
    else:
        plot(converted_Y_t, converted_predicted_Y, 'Actual Y_t', 'Predicted Y_t', "Actual vs Predicted")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(config["dataset"]["path"], 
                    config["dataset"]["brand"], 
                    config["dataset"]["dependent_variable"], 
                    config["dataset"]["independent_variables_X"], 
                    config["dataset"]["independent_variables_Z"])

    X_t, Z_t, Y_t = data_preprocessing.preprocess(normalization=True)
    main(X_t, Z_t, Y_t, do_data_simulation=False)

    # #compare between simulated and predicted data
    # data_simulation = DataSimulation(X_t, Z_t, Y_t, config["modelTraining"]["originalG"], config["modelTraining"]["theta0"])
    # simulated_results = data_simulation.get_simulation_results()
    
    # if config['inferenceMethod'] == "torch_autograd":
    #     X_t = torch.tensor(X_t, dtype=torch.float32)
    #     Z_t = torch.tensor(Z_t, dtype=torch.float32)
    #     Y_t = torch.tensor(Y_t, dtype=torch.float32)

    # Trainer(X_t, Z_t, simulated_results['simulated_Y']).train()
    # predicted_simulated_Y= SalesPrediction().predict(X_t, Z_t)
    
    # _, _, scaler_Y = data_preprocessing.get_normalized_scaler()
    # converted_simulated_Y = convert_normalized_data(scaler_Y, simulated_results['simulated_Y'])
    # converted_predicted_Y = convert_normalized_data(scaler_Y, predicted_simulated_Y)

    # plot(simulated_results['actual_Y'], simulated_results['simulated_Y'], 'Actual Y_t', 'Simulated Y_t',  "simulated vs actual")
    # plot(converted_simulated_Y, converted_predicted_Y, 'Simulated Y_t', 'Predicted Y_t', "simulated vs predicted")
    # plt.tight_layout()
    # plt.show()

    # # training data from the scratch, compare the predicted data with the actual one
    # Trainer(X_t, Z_t, Y_t).train()





    # TODO: 
    # prepare for csi (top priority)
    # adum 
    # initialization of parameters and optimization of parameters in gd
    # quantify the long term effect of the marketing campaign on sales
    # G should be learned
    # param recovery
 