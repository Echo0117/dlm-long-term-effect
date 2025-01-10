# dynamic_linear_model/experiments/simulation_experiment.py

import os
import pickle
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from tqdm import tqdm
from loguru import logger
from datetime import datetime
import os
from ray.train import report


import math




# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)


from config import config
from dynamic_linear_model.model import DynamicLinearModel
from dynamic_linear_model.losses import mse_loss
import dynamic_linear_model.utils as utils
from dynamic_linear_model.data_processing import DataPreprocessing

device = config["device"]

def train_model(config_inner, X_t, Z_t, Y_t, folder_name):
    """
    Training function to be used with Ray Tune.
    """
    # Ensure inputs are tensors
    if isinstance(X_t, np.ndarray):
        X_t = torch.tensor(X_t, dtype=torch.float32, device=device)
    if isinstance(Z_t, np.ndarray):
        Z_t = torch.tensor(Z_t, dtype=torch.float32, device=device)
    if isinstance(Y_t, np.ndarray):
        Y_t = torch.tensor(Y_t, dtype=torch.float32, device=device)

    # Initialize the model
    model = DynamicLinearModel(config["simulationRecovery"]["independentRun"]).to(device)

    # # Define optimizer with separate parameter groups
    # optimizer = optim.Adam([
    #     {'params': model.gamma, 'lr': config_inner["training.gamma_lr"], 'weight_decay': config_inner["training.gamma_weight_decay"]},
    #     {'params': model.zeta, 'lr': config_inner["training.zeta_lr"], 'weight_decay': config_inner["training.zeta_weight_decay"]},
    #     {'params': model.eta, 'lr': config_inner["training.eta_lr"], 'weight_decay': config_inner["training.eta_weight_decay"]},
    #     # Add other parameter groups if necessary
    # ])

    optimizer = optim.Adam(
        # self.model.parameters(),
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config_inner["training.gamma_lr"],
        weight_decay=config_inner["training.gamma_weight_decay"],
    )
    # Define scheduler with tunable parameters
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config_inner.get("scheduler.factor", 0.5),
        patience=config_inner.get("scheduler.patience", 10),
        threshold=config_inner.get("scheduler.threshold", 0.0001),  # map min_delta to threshold
        verbose=False  # Suppress deprecated warning
    )

    epoch_limit = config["modelTraining"]["epoch"]
    early_stopping_patience = config_inner.get("early_stopping.patience", config["modelTraining"]["patience"])
    early_stopping_min_delta = config_inner.get("early_stopping.min_delta", config["modelTraining"]["minDelta"])

    best_loss = np.inf
    early_stop_counter = 0

    for epoch in tqdm(range(epoch_limit), desc="Ray Tune Training"):
        model.train()
        optimizer.zero_grad()
        outputs = model.forward(X_t, Z_t)
        # loss = mse_loss(outputs, Y_t.repeat(config["simulationRecovery"]["independentRun"], 1)) + (model.gamma.pow(2).mean() * config_inner.get("training.l2_lambda", 1e-4))
        # loss = loss.mean() + (model.gamma.pow(2).mean() * config_inner.get("training.l2_lambda", 1e-4))
        
        # loss = mse_loss(outputs, Y_t.repeat(5, 1)) + model.gamma.pow(2).mean() * config_inner.get("training.l2_lambda", 1e-4)
        gamma_l2_penalty = model.gamma.pow(2).mean() * config_inner.get("training.l2_lambda", 1e-4)
        loss = mse_loss(outputs, Y_t.repeat(config["simulationRecovery"]["independentRun"], 1)) 

        # loss = loss.mean()  # Aggregate loss across runs and time steps
        # loss.backward()
        
        # Backward each loss separately
        for l in loss:
            loss_penalty = l + gamma_l2_penalty
            loss_penalty.backward(retain_graph=True)
        loss = loss.mean() 

        # Gradient clipping
        max_grad_norm = config["modelTraining"]["maxGradNorm"]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Scheduler step based on training loss
        scheduler.step(loss)

        # Early stopping based on training loss
        current_loss = loss.item()
        if best_loss - current_loss > early_stopping_min_delta:
            best_loss = current_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Report to Ray Tune
        report(dict(loss=current_loss))

        if early_stop_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Optionally, save the trained model
    # torch.save(model.state_dict(), os.path.join(folder_name, "best_model.pth"))
    return best_loss

def trainer_with_ray_tune(X_t, Z_t, Y_t, folder_name):
    """
    Trainer function that integrates Ray Tune for hyperparameter tuning.
    """
    # Define the search space
    search_space = {
        "training.gamma_lr": tune.loguniform(1e-5, 1e-2),
        "training.gamma_weight_decay": tune.loguniform(1e-5, 1e-2),
        # "training.zeta_lr": tune.loguniform(1e-5, 1e-2),
        # "training.zeta_weight_decay": tune.loguniform(1e-5, 1e-2),
        # "training.eta_lr": tune.loguniform(1e-5, 1e-2),
        # "training.eta_weight_decay": tune.loguniform(1e-5, 1e-2),
        "training.max_grad_norm": tune.uniform(0.1, 5.0),
        "scheduler.patience": tune.choice([5, 10, 15, 20]),
        "scheduler.threshold": tune.loguniform(1e-5, 1e-2),
        "scheduler.factor": tune.loguniform(0.1, 0.9),
        "early_stopping.patience": tune.choice([5, 10, 15, 20]),
        "early_stopping.min_delta": tune.loguniform(1e-5, 1e-2),
        "training.l2_lambda": tune.loguniform(1e-5, 1e-2)
    }

    # Set up the ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=config["modelTraining"]["epoch"],
        grace_period=10,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=[
            "training.gamma_lr", "training.gamma_weight_decay",
            # "training.zeta_lr", "training.zeta_weight_decay",
            # "training.eta_lr", "training.eta_weight_decay",
            "training.max_grad_norm",
            "training.l2_lambda",
            "scheduler.patience", "scheduler.threshold",
            "scheduler.factor",
            "early_stopping.patience", "early_stopping.min_delta"
        ],
        metric_columns=["loss", "training_iteration"]
    )

    # Define a function to pass to Ray Tune that includes the data
    def tune_train(config_inner):
        try:
            loss = train_model(config_inner, X_t, Z_t, Y_t, folder_name)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            report(dict(loss=float("inf")))

    # Run Ray Tune
    analysis = tune.run(
        tune.with_parameters(tune_train),
        resources_per_trial={"cpu": 8},  # Adjust based on your resources
        config=search_space,
        num_samples=config["simulationRecovery"]["independentRun"],
        scheduler=scheduler,
        progress_reporter=reporter,
        name="ray_tune_simulation_recovery"
    )

    # Get the best trial
    best_trial = analysis.get_best_trial("loss", "min", "last")
    best_config = best_trial.config
    logger.info(f"Best config: {best_config}")

    # Save the best configuration
    with open(os.path.join(folder_name, "best_config.pkl"), "wb") as f:
        pickle.dump(best_config, f)

    # Optionally, save the analysis results
    results_df = analysis.results_df
    results_df.to_csv(os.path.join(folder_name, "tune_results.csv"), index=False)

    # Return the best hyperparameters
    return best_config


if __name__ == "__main__":
    # Determine the number of brands and calculate a suitable grid size (rows and columns)
    num_brands = len(config["dataset"]["brand_list"])
    cols = min(4, num_brands)  # Limit columns to 4 for readability
    rows = math.ceil(num_brands / cols)  # Calculate rows based on number of columns

    # Initialize lists to accumulate gamma and zeta values across brands
    all_gamma_best = []
    all_zeta_best = []
    all_brands = []
    all_max_grad_norm_best = []
    all_l2_lambda_best = []

    for i, brand in enumerate(config["dataset"]["brand_list"]):
        print("Processing brand:", brand)
        config["dataset"]["brand"] = brand

        # Create unique folder names for each brand
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        folder_name = f"""dataset/generated_results/train_results/multiple_brands/epoch{config["modelTraining"]["epoch"]}_run{config["simulationRecovery"]["independentRun"]}_{config["dataset"]["brand"]}_{timestamp}"""
        os.makedirs(folder_name, exist_ok=True)

        # Update paths in configuration
        config["modelTraining"]["lossPath"] = f"{folder_name}/loss_results.csv"
        config["modelTraining"]["parametersPath"] = f"{folder_name}/parameters_results.csv"
        config["simulationRecovery"]["paramsSavedPath"] = f"{folder_name}/simulated_parameters.csv"
        config["simulationRecovery"]["simulatedParamSavedPath"] = f"{folder_name}/temp_simulated.csv"

        # Initialize DataPreprocessing with the updated brand
        data_preprocessing = DataPreprocessing(
            config["dataset"]["path"],
            brand,
            config["dataset"]["dependent_variable"],
            config["dataset"]["independent_variables_X"],
            config["dataset"]["independent_variables_Z"],
        )

        # Preprocess the data
        X_t, Z_t, Y_t = data_preprocessing.preprocess(normalization=True)

        # Ensure data is in tensor format
        if isinstance(X_t, np.ndarray):
            X_t = torch.tensor(X_t, dtype=torch.float32, device=config["device"])
        if isinstance(Z_t, np.ndarray):
            Z_t = torch.tensor(Z_t, dtype=torch.float32, device=config["device"])
        if isinstance(Y_t, np.ndarray):
            Y_t = torch.tensor(Y_t, dtype=torch.float32, device=config["device"])

        # Run the simulation recovery with Ray Tune
        best_config = trainer_with_ray_tune(X_t, Z_t, Y_t, folder_name)

        # Accumulate gamma and zeta values across brands for combined analysis
        # Assuming that best_config contains the necessary information
        # all_gamma_best.append(best_config["training.gamma_lr"])
        # all_zeta_best.append(best_config["training.zeta_lr"])
        all_brands.append(brand)
        all_max_grad_norm_best.append(best_config["training.max_grad_norm"])
        all_l2_lambda_best.append(best_config["training.l2_lambda"])

        # Log the best hyperparameters
        logger.info(f"Brand: {brand}, Best Config: {best_config}")

    # Optionally, save the accumulated results or perform further analysis
    # Since plotting is not required, we can skip it
    # For demonstration, let's save the best hyperparameters
    best_hyperparams_df = pd.DataFrame({
        "Brand": all_brands,
        # "Best_gamma_lr": all_gamma_best,
        # "Best_zeta_lr": all_zeta_best,
        "Best_max_grad_norm": all_max_grad_norm_best,
        "Best_l2_lambda": all_l2_lambda_best,
        # Add other hyperparameters if needed
    })
    best_hyperparams_df.to_csv("dataset/generated_results/train_results/best_hyperparameters.csv", index=False)

    print("Ray Tune hyperparameter tuning completed for all brands.")