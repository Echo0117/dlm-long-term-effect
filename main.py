import logging
from config import config
from dynamic_linear_model.data_processing import DataPreprocessing
from dynamic_linear_model.experiments.simulation_experiment import simulation_recovery, simulation_recovery_with_multi_independent_runs

# Configure logging
logging.basicConfig(
    filename="log.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(
        config["dataset"]["path"],
        config["dataset"]["brand"],
        config["dataset"]["dependent_variable"],
        config["dataset"]["independent_variables_X"],
        config["dataset"]["independent_variables_Z"],
    )

    X_t, Z_t, Y_t = data_preprocessing.preprocess(normalization=True)

    # simulation_recovery(X_t, Z_t, Y_t)

    simulation_recovery_with_multi_independent_runs(X_t, Z_t, Y_t)
