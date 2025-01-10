import logging
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger
from matplotlib.axes import Axes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from tqdm import tqdm
import ipdb
from config import config
from dynamic_linear_model.losses import mse_loss
from dynamic_linear_model.model import DynamicLinearModel
import dynamic_linear_model.utils as utils


device = config["device"]


class SimulationRecovery:
    def __init__(self, X_t: np.ndarray, Z_t: np.ndarray, Y_t: np.ndarray):
        self.X_t = torch.tensor(X_t, dtype=torch.float32, device=device)
        self.Z_t = torch.tensor(Z_t, dtype=torch.float32, device=device)
        self.Y_t = torch.tensor(Y_t, dtype=torch.float32, device=device)
        self.epoch = config["modelTraining"]["epoch"]
        self.num_runs = config["simulationRecovery"]["independentRun"]

        # Initialize the model with parameters for all runs
        self.model = DynamicLinearModel(self.num_runs).to(device)

    def recovery_for_simulation(
        self, ax: Axes
    ) -> np.ndarray:

        plotter = utils.Plotter()
        (
            named_parameters,
            best_epoch,
            best_loss_run_index,
            best_loss,
            best_predicted_Y,
            val_best_predicted_Y
        ) = self._optimize(self.Y_t[:240], self.X_t[:240], self.Z_t[:240], self.Y_t[240:], self.X_t[240:], self.Z_t[240:])
        logger.info(
            f"best final loss: {best_loss}, best final loss is at epoch: {best_epoch}, the index of best run: {best_loss_run_index}"
        )

        recovered_parameters = plotter.plot_params(named_parameters, ax)
        utils.calculate_statistics(recovered_parameters, best_loss_run_index)
        val_indices = range(240, len(self.X_t))
        return val_best_predicted_Y, val_indices

    def _optimize(
        self, Y_t: torch.Tensor, X_t: torch.Tensor, Z_t: torch.Tensor,
        Y_val: torch.Tensor = None, X_val: torch.Tensor = None, Z_val: torch.Tensor = None,
    ) -> tuple:
        # optimizer = optim.SGD(
        #     self.model.parameters(),
        #     lr=config["modelTraining"]["learningRate"],
        #     momentum=config["modelTraining"]["momentum"],
        #     weight_decay=config["modelTraining"]["weightDecay"],
        # )

        optimizer = optim.Adam(
            # self.model.parameters(),
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config["modelTraining"]["learningRate"],
            weight_decay=config["modelTraining"]["weightDecay"],
        )

        # # Suppose you want to update gamma more slowly and without L2 regularization
        # gamma_lr = 1e-3
        # gamma_weight_decay =  1e-1

        # # Update zeta with a standard learning rate and some L2 regularization
        # zeta_lr = 1e-3
        # zeta_weight_decay = 0.0

        # # Other parameters use default settings
        # default_lr = 1e-3
        # default_weight_decay = 1e-5

        # other_params = [param for name, param in self.model.named_parameters()
        #         if name not in ['gamma', 'zeta'] and param.requires_grad]

        # # Initialize the optimizer with parameter groups
        # optimizer = optim.Adam([
        #     {'params': self.model.gamma, 'lr': gamma_lr, 'weight_decay': gamma_weight_decay},
        #     {'params': self.model.zeta, 'lr': gamma_lr, 'weight_decay': zeta_weight_decay},
        #     {'params': other_params, 'lr': gamma_lr, 'weight_decay': default_weight_decay}
        # ])

        (
            outputs_list,
            losses_list,
            all_parameters_list,
            all_parameters_dict,
        ) = ([], [], [], {})

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config["modelTraining"]["factor"], patience=config["modelTraining"]["patience"], verbose=True)

        val_losses = []
        val_outputs_list = []
        early_stop_counter = 0
        best_val_loss = np.inf
        best_epoch = 0
        val_loss = None
        gamma_l2_lambda = config["modelTraining"]["l2Lambda"]
        for epoch in tqdm(range(self.epoch), desc="Epochs"):

            optimizer.zero_grad()
            outputs = self.model.forward(X_t, Z_t)
            # method 1 L2 penalty on gamma
            # batch_losses = mse_loss(outputs, Y_t.repeat(self.num_runs, 1)) + self.model.gamma.pow(2).mean() * config["modelTraining"].get("l2Lambda", 1e-4)
            
            # method 2 L2 penalty on gamma
            batch_losses = mse_loss(outputs, Y_t.repeat(self.num_runs, 1))
            gamma_l2_penalty = self.model.gamma.pow(2).mean() * gamma_l2_lambda

            # Backward each loss separately
            for loss in batch_losses:
                loss_penalty = loss + gamma_l2_penalty
                loss_penalty.backward(retain_graph=True)

            # TODO:
            # 2-5% after converge
            # Lower the lr, train longer
            # the converge of the paramaters, 

            # Clip gradients to avoid exploding gradients
            # max_grad_norm = config["modelTraining"]["maxGradNorm"]
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            max_grad_value = config["modelTraining"].get("maxGradValue", 0.5) 
            torch.nn.utils.clip_grad_value_(self.model.parameters(), max_grad_value)

            optimizer.step()

            params_after_optim = {
                name: param.clone().detach().cpu()
                for name, param in self.model.named_parameters()
            }

            all_parameters_list.append(params_after_optim)  

            # Validation phase
            if X_val is not None and Y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model.forward(X_val, Z_val)
                    val_batch_losses = mse_loss(val_outputs, Y_val.repeat(self.num_runs, 1))
                    val_loss = val_batch_losses.mean().item()
                    val_losses.append(val_loss)

                # Early stopping check
                if best_val_loss - val_loss > config["modelTraining"]["minDelta"]:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= config["modelTraining"]["patience"]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            losses_list.append(batch_losses.detach().clone())
            outputs_list.append(outputs.detach().clone())
            val_outputs_list.append(val_outputs.detach().clone())

            # logger at specified intervals
            if epoch % 100 == 0 or epoch == epoch - 1:
                if val_loss is not None:
                    logger.info(
                        f"Epoch {epoch}: Training Loss = {batch_losses.mean().item():.6f}, gamma_l2_penalty = {gamma_l2_penalty:.6f}, Validation Loss = {val_loss:.6f}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch}: Training Loss = {batch_losses.mean().item():.6f}"
                    )
            
            # Step the learning rate scheduler
            #TODO: apply scheduler on Training loss 
            # if val_loss is not None:
            #     scheduler.step(val_loss)

            mean_train_loss = batch_losses.mean().item()
            if mean_train_loss is not None:
                scheduler.step(mean_train_loss)

        losses_tensor = torch.stack(losses_list)  # Shape: (epoch, num_runs, T)
        outputs_tensor = torch.stack(outputs_list)  # Shape: (epoch, num_runs, T)
        losses_np = losses_tensor.cpu().numpy()  # Move once to CPU and convert
        outputs_np = outputs_tensor.cpu().numpy()

        val_outputs_tensor = torch.stack(val_outputs_list) 
        val_outputs_np = val_outputs_tensor.cpu().numpy()

        # Find the index of the index of best performance
        min_index_1d = np.argmin(losses_np)
        min_index_2d = np.unravel_index(min_index_1d, losses_np.shape)

        best_epoch = min_index_2d[0]
        best_loss_run_index = min_index_2d[1]
        best_loss = losses_np[min_index_2d]
        best_predicted_Y = outputs_np[min_index_2d]
        val_best_predicted_Y = val_outputs_np[min_index_2d]

        # Create a DataFrame with the required structure
        min_indices_by_column_independent = np.argmin(losses_np, axis=0)
        min_values_by_column_independent = losses_np[
            min_indices_by_column_independent, np.arange(losses_np.shape[1])
        ]
        loss_data = {
            "independent run": [
                x + 1 for x in list(range(len(min_indices_by_column_independent)))
            ],
            "epoch index": min_indices_by_column_independent,
            "loss value": min_values_by_column_independent,
        }
        pd.DataFrame(loss_data).to_csv(config["modelTraining"]["lossPath"])
        logger.info(f"file loss_results saved at {config['modelTraining']['lossPath']}")

        # Convert the generator to a list of dictionaries
        param_list = list(all_parameters_list)

        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(param_list)
        df.columns = ["G", "eta", "zeta", "gamma"]
        df.to_csv(config["modelTraining"]["parametersPath"], index=False)

        return (
            self.model.named_parameters(),
            best_epoch,
            best_loss_run_index,
            best_loss,
            best_predicted_Y,
            val_best_predicted_Y
        )

    # def cross_validate(self, k_folds=5) -> float:
    #     tscv = TimeSeriesSplit(n_splits=k_folds)
    #     all_y_trues = []
    #     all_y_preds = []
    #     val_losses = []
    #     val_r_squared_scores = []

    #     for fold_index, (train_indices, val_indices) in enumerate(tscv.split(self.X_t)):
    #         logger.info(f"Fold {fold_index + 1}/{k_folds}")

    #         # Split the data into training and validation sets
    #         # Further split the training indices for early stopping validation
    #         train_size = int(len(train_indices) * 0.8)
    #         train_idx = train_indices[:train_size]
    #         val_idx = train_indices[train_size:]

    #         X_train, X_val = self.X_t[train_idx], self.X_t[val_idx]
    #         Z_train, Z_val = self.Z_t[train_idx], self.Z_t[val_idx]
    #         Y_train, Y_val = self.Y_t[train_idx], self.Y_t[val_idx]
    #         splits = list(tscv.split(self.X_t))
    #         print(f"Number of splits: {len(splits)}")
    #         #print("train_size", train_size)
    #         print("Y_trainlene", len(Y_train))
    #         print("Y_vallene", len(Y_val))
    #         start = time.time()
    #         # Initialize and train the model with early stopping
    #         (_, _, _, _, best_predicted_Y) = self._optimize(Y_train, X_train, Z_train, Y_val, X_val, Z_val)
    #         end = time.time()
    #         print("cross v time", end-start)
    #         # Evaluate on test validation data (val_indices from TimeSeriesSplit
    #         with torch.no_grad():
    #             X_val_fold, Z_val_fold, Y_val_fold = (
    #                 self.X_t[val_indices],
    #                 self.Z_t[val_indices],
    #                 self.Y_t[val_indices],
    #             )
    #             outputs = self.model.forward(X_val_fold, Z_val_fold)
    #             batch_losses = mse_loss(outputs, Y_val_fold.repeat(self.num_runs, 1))
    #             val_loss = batch_losses.mean().item()
    #             val_losses.append(val_loss)

    #             # Compute R-squared
    #             y_true = Y_val_fold.cpu().numpy()
    #             y_pred = outputs.mean(dim=0).cpu().numpy()  # Average over runs
    #             all_y_trues.append(y_true)
    #             all_y_preds.append(y_pred)

    #             ss_res = np.sum((y_true - y_pred) ** 2)
    #             ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    #             r_squared = 1 - (ss_res / ss_tot)
    #             val_r_squared_scores.append(r_squared)

    #             logger.info(f"Validation Loss for fold {fold_index + 1}: {val_loss}")
    #             logger.info(f"R-squared for fold {fold_index + 1}: {r_squared}")

    #     avg_val_loss = sum(val_losses) / len(val_losses)
    #     avg_r_squared = sum(val_r_squared_scores) / len(val_r_squared_scores)

    #     logger.info(f"Average validation loss across {k_folds} folds: {avg_val_loss:.3f}")
    #     logger.info(f"Average R-squared loss across {k_folds} folds: {avg_r_squared:.3f}")

    #     val_losses_formatted = [f"{val:.3f}" for val in val_losses]
    #     r_squared_scores_formatted = [f"{r2:.3f}" for r2 in val_r_squared_scores]

    #     logger.info(f"List validation loss across {k_folds} folds: {val_losses_formatted}")
    #     logger.info(f"List R-squared across {k_folds} folds: {r_squared_scores_formatted}")

    #     # Concatenate all true and predicted Y values
    #     all_y_trues = np.concatenate(all_y_trues)
    #     all_y_preds = np.concatenate(all_y_preds)

    #     # Compute overall R-squared
    #     ss_res = np.sum((all_y_trues - all_y_preds) ** 2)
    #     ss_tot = np.sum((all_y_trues - np.mean(all_y_trues)) ** 2)
    #     overall_r_squared = 1 - (ss_res / ss_tot)
    #     logger.info(f"Overall R-squared across all folds: {overall_r_squared:.3f}")

    #     return best_predicted_Y
    def cross_validate(self, k_folds=5) -> float:
        """
        Perform cross-validation with time series data using TimeSeriesSplit.

        Args:
            k_folds (int): Number of folds for cross-validation.

        Returns:
            float: The best-predicted Y values from the best fold.
        """
        tscv = TimeSeriesSplit(n_splits=k_folds)
        all_y_trues = []
        all_y_preds = []
        val_losses = []
        val_r_squared_scores = []

        # Iterate through each split from TimeSeriesSplit
        for fold_index, (train_indices, val_indices) in enumerate(tscv.split(self.X_t)):
            logger.info(f"Fold {fold_index + 1}/{k_folds}")

            # Use all training indices provided by TimeSeriesSplit
            X_train, X_val_fold = self.X_t[train_indices], self.X_t[val_indices]
            Z_train, Z_val_fold = self.Z_t[train_indices], self.Z_t[val_indices]
            Y_train, Y_val_fold = self.Y_t[train_indices], self.Y_t[val_indices]

            # Debugging logs for train and validation set sizes
            logger.info(f"Train size: {len(train_indices)}")
            logger.info(f"Validation size: {len(val_indices)}")

            # Time tracking for optimization
            start = time.time()

            # Optimize model using the training data
            (_, _, _, _, best_predicted_Y, val_best_predicted_Y) = self._optimize(Y_train, X_train, Z_train, Y_val_fold, X_val_fold, Z_val_fold)

            end = time.time()
            logger.info(f"Cross-validation fold {fold_index + 1} time: {end - start:.2f} seconds")

            # Evaluate on the validation data for the current fold
            with torch.no_grad():
                outputs = self.model.forward(X_val_fold, Z_val_fold)
                batch_losses = mse_loss(outputs, Y_val_fold.repeat(self.num_runs, 1))
                val_loss = batch_losses.mean().item()
                val_losses.append(val_loss)

                # Compute R-squared
                y_true = Y_val_fold.cpu().numpy()
                y_pred = outputs.mean(dim=0).cpu().numpy()  # Average over runs
                all_y_trues.append(y_true)
                all_y_preds.append(y_pred)

                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                val_r_squared_scores.append(r_squared)

                logger.info(f"Validation Loss for fold {fold_index + 1}: {val_loss}")
                logger.info(f"R-squared for fold {fold_index + 1}: {r_squared}")

        # Compute average metrics across all folds
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_r_squared = sum(val_r_squared_scores) / len(val_r_squared_scores)

        logger.info(f"Average validation loss across {k_folds} folds: {avg_val_loss:.3f}")
        logger.info(f"Average R-squared loss across {k_folds} folds: {avg_r_squared:.3f}")

        val_losses_formatted = [f"{val:.3f}" for val in val_losses]
        r_squared_scores_formatted = [f"{r2:.3f}" for r2 in val_r_squared_scores]

        logger.info(f"List of validation losses across {k_folds} folds: {val_losses_formatted}")
        logger.info(f"List of R-squared values across {k_folds} folds: {r_squared_scores_formatted}")

        # Concatenate all true and predicted Y values across folds
        all_y_trues = np.concatenate(all_y_trues)
        all_y_preds = np.concatenate(all_y_preds)

        # Compute overall R-squared across all folds
        ss_res = np.sum((all_y_trues - all_y_preds) ** 2)
        ss_tot = np.sum((all_y_trues - np.mean(all_y_trues)) ** 2)
        overall_r_squared = 1 - (ss_res / ss_tot)
        logger.info(f"Overall R-squared across all folds: {overall_r_squared:.3f}")
        # Return best-predicted Y values for the best fold
        return val_best_predicted_Y, val_indices


    def get_predicted_Y(self, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        predicted_Y = self.model(X, Z).detach()
        return predicted_Y


# class DataSimulation:
#     def __init__(
#         self,
#         X_t: np.ndarray,
#         Z_t: np.ndarray,
#         Y_t: np.ndarray,
#         G: float = 0,
#     ):
#         """
#         Initialize the DataSimulation class with data and parameters.

#         Parameters:
#         X_t (np.ndarray): Independent variables matrix X.
#         Z_t (np.ndarray): Independent variables matrix Z.
#         Y_t (np.ndarray): Dependent variable vector Y.
#         G (float, optional): State transition coefficient. Default is 0.
#         """
#         self.X_t = torch.tensor(X_t, dtype=torch.float32, device=device)
#         self.Z_t = torch.tensor(Z_t, dtype=torch.float32, device=device)
#         self.Y_t = torch.tensor(Y_t, dtype=torch.float32, device=device)
#         self.G = nn.Parameter(torch.tensor(G, device=device))
#         self._obtain_initial_parameters_by_lr()
#         self.save_simulated_parameters()
#         self.model = DynamicLinearModel(num_runs=1).to(device)

#     def generate_simulated_Y(self) -> torch.Tensor:
#         """
#         Apply the Dynamic Linear Model (DLM) to predict values.

#         Returns:
#         torch.Tensor: Predicted values.
#         """
#         self.model.G = self.G
#         self.model.eta = self.eta
#         self.model.zeta = self.zeta
#         self.model.gamma = self.gamma
#         simulated_Y = self.model.forward(
#             self.X_t, self.Z_t, is_simulation=True
#         )
#         return simulated_Y[0]

#     def get_simulation_results(self) -> dict:
#         """
#         Get a dictionary with actual and predicted Y values for visualization.

#         Returns:
#         dict: Dictionary containing actual and predicted Y values.
#         """
#         simulated_Y = self.generate_simulated_Y().detach().cpu().numpy()
#         results_Y = {"actual_Y": self.Y_t.cpu().numpy(), "simulated_Y": simulated_Y}
#         utils.log_parameters_results(self.G, self.eta, self.zeta)
#         return results_Y

#     def save_simulated_parameters(self):
#         """
#         Save the simulated parameters to a CSV file by adding new rows above the original file.
#         """
#         simulated_params = {
#             "Param": ["G"]
#             + [f"η{i+1}" for i in range(len(self.eta))]
#             + [f"ζ{i+1}" for i in range(len(self.zeta))]
#             + [f"γ{i+1}" for i in range(len(self.gamma))],
#             "Simulated": [self.G.data.cpu().numpy()]
#             + list(self.eta.data.cpu().numpy())
#             + list(self.zeta.data.cpu().numpy())
#             + list(self.gamma.data.cpu().numpy()),
#         }
#         df_simulated = pd.DataFrame(simulated_params)

#         file_path = config["simulationRecovery"]["simulatedParamSavedPath"]

#         if os.path.exists(file_path):
#             existing_df = pd.read_csv(file_path)
#             combined_df = pd.concat([df_simulated, existing_df], ignore_index=True)
#         else:
#             combined_df = df_simulated

#         combined_df.to_csv(file_path, index=False)

#     def _obtain_initial_parameters_by_lr(self):
#         """
#         Obtain eta (η), zeta (ζ) gamma (γ) by fitting a linear regression model to XZ_t and Y_t.
#         """
#         XZ_t = np.hstack((self.X_t.cpu().numpy(), self.Z_t.cpu().numpy()))
#         model_XZ = LinearRegression(
#             fit_intercept=False,
#             # positive=True
#         ).fit(XZ_t, self.Y_t.cpu().numpy())
#         eta_zeta = model_XZ.coef_

#         self.eta = nn.Parameter(
#             torch.tensor(eta_zeta[: self.X_t.shape[1]], device=device)
#         )
#         self.zeta = nn.Parameter(
#             torch.tensor(eta_zeta[self.X_t.shape[1] :], device=device)
#         )

#         if config["simulationRecovery"]["isGammaEqualZeta"]:
#             self.gamma = self.zeta
#         else:
#             self.gamma = nn.Parameter(
#                 torch.randn(config["dataset"]["zDim"], device=device)
#             )
