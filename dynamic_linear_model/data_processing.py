import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from config import config

class DataPreprocessing:
    def __init__(
        self,
        file_path: str,
        brand: str,
        dependent_variable: str,
        independent_variables_X: List[str],
        independent_variables_Z: List[str],
    ):
        """
        Initialize the DataPreprocessing class with the necessary parameters.

        Parameters:
        file_path (str): Path to the CSV file containing the data.
        brand (str): Brand name to filter the data.
        dependent_variable (str): The dependent variable column name.
        independent_variables_X (List[str]): List of column names for the independent control variables X.
        independent_variables_Z (List[str]): List of column names for the independent interested variables Z.
        """
        self.file_path = file_path
        self.brand = brand
        self.dependent_variable = dependent_variable
        self.independent_variables_X = independent_variables_X
        self.independent_variables_Z = independent_variables_Z
        self.scaler_X = AbsoluteMedianScaler()
        self.scaler_Z = AbsoluteMedianScaler()
        self.scaler_Y = AbsoluteMedianScaler()

    def preprocess(
        self, normalization: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data by loading, filtering, extracting variables, and optionally normalizing them.

        Parameters:
        normalization (bool): Flag to indicate whether to normalize the data.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the normalized (or original) X_t, Z_t, and Y_t.
        """
        df = self._load_data()
        df = self._filter_brand(df)
        Y_t = self._dependent_variable(df)
        X_t, Z_t = self._independent_variables(df)

        # Handle missing values by replacing NaNs with zeros
        X_t, Z_t, Y_t = self._handle_missing_values(X_t, Z_t, Y_t)
        X_t, Z_t, Y_t, non_zero_indices = self._remove_zero_y_values(X_t, Z_t, Y_t)
        config["dataset"]["year_week"] = df["year_week"].values[non_zero_indices]

        if normalization:
            X_t, Z_t, Y_t = self._normalize_data(X_t, Z_t, Y_t)

        return X_t, Z_t, Y_t

    def get_normalized_scaler(self):
        """
        Get the scalers used for normalization.

        Returns:
        Tuple[AbsoluteMedianScaler, AbsoluteMedianScaler, AbsoluteMedianScaler]: Scalers for X, Z, and Y.
        """
        return self.scaler_X, self.scaler_Z, self.scaler_Y

    def _load_data(self) -> pd.DataFrame:
        """
        Load the data from a CSV file.

        Returns:
        pd.DataFrame: DataFrame containing the loaded data.
        """
        df = pd.read_csv(self.file_path, sep=";")
        return df

    def _filter_brand(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the data by the specified brand.

        Parameters:
        df (pd.DataFrame): DataFrame containing the data.

        Returns:
        pd.DataFrame: Filtered DataFrame containing only the specified brand.
        """
        df = df[df["brand"] == self.brand]
        return df

    def _dependent_variable(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract the dependent variable from the data.

        Parameters:
        df (pd.DataFrame): DataFrame containing the data.

        Returns:
        np.ndarray: Array containing the dependent variable values.
        """
        Y_t = df[self.dependent_variable].values
        return Y_t

    def _independent_variables(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the independent variables X and Z from the data.

        Parameters:
        df (pd.DataFrame): DataFrame containing the data.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the arrays for independent variables X and Z.
        """
        X_t = df[self.independent_variables_X].values
        Z_t = df[self.independent_variables_Z].values
        return X_t, Z_t

    def _normalize_data(
        self, X_t: np.ndarray, Z_t: np.ndarray, Y_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize the independent and dependent variables.

        Parameters:
        X_t (np.ndarray): Array containing the independent variables X.
        Z_t (np.ndarray): Array containing the independent variables Z.
        Y_t (np.ndarray): Array containing the dependent variable values.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the normalized X_t, Z_t, and Y_t.
        """
        if config["dataset"]["isNormalizeX"]:
            X_t_normalized = self.scaler_X.fit_transform(X_t)
        else:
            X_t_normalized = X_t
        Z_t_normalized = self.scaler_Z.fit_transform(Z_t)
        Y_t_normalized = self.scaler_Y.fit_transform(Y_t.reshape(-1, 1)).flatten()

        return X_t_normalized, Z_t_normalized, Y_t_normalized
    
    def _handle_missing_values(
        self, X_t: np.ndarray, Z_t: np.ndarray, Y_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Handle missing values in X_t, Z_t, and Y_t by replacing NaNs with zeros.
        
        Parameters:
        X_t (np.ndarray): Array containing the independent variables X.
        Z_t (np.ndarray): Array containing the independent variables Z.
        Y_t (np.ndarray): Array containing the dependent variable values.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays with NaNs replaced by zeros.
        """
        # Replace NaNs with zeros
        X_t = np.nan_to_num(X_t, nan=0.0)
        Z_t = np.nan_to_num(Z_t, nan=0.0)
        Y_t = np.nan_to_num(Y_t, nan=0.0)

        return X_t, Z_t, Y_t

    def _remove_zero_y_values(
        self, X_t: np.ndarray, Z_t: np.ndarray, Y_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove rows where Y_t has a value of zero.

        Parameters:
        X_t (np.ndarray): Array containing the independent variables X.
        Z_t (np.ndarray): Array containing the independent variables Z.
        Y_t (np.ndarray): Array containing the dependent variable values.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered arrays with rows removed where Y_t is zero.
        """
        non_zero_indices = Y_t != 0  # Create a mask for rows where Y_t is non-zero
        X_t = X_t[non_zero_indices]
        Z_t = Z_t[non_zero_indices]
        Y_t = Y_t[non_zero_indices]
        return X_t, Z_t, Y_t, non_zero_indices

class AbsoluteMedianScaler(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'AbsoluteMedianScaler':
        """
        Fit the scaler by calculating the median of each feature.

        Parameters:
        X (np.ndarray): The input data to fit.
        y (np.ndarray, optional): The target values (ignored).

        Returns:
        AbsoluteMedianScaler: The fitted scaler.
        """
        self.medians_ = []
        for col in range(X.shape[1]):
            col_median = np.median(X[:, col])
            if col_median == 0:
                non_zero_values = X[:, col][X[:, col] != 0]
                if non_zero_values.size > 0:
                    col_median = np.median(non_zero_values)
                else:
                    col_median = 1  # Default value if all values are zero

            self.medians_.append(np.abs(col_median))

        self.medians_ = np.array(self.medians_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale the data by the absolute value of the median of each column.

        Parameters:
        X (np.ndarray): The input data to transform.

        Returns:
        np.ndarray: The transformed data.
        """
        return X / self.medians_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale back the data by multiplying with the median of each column.

        Parameters:
        X (np.ndarray): The transformed data to inverse.

        Returns:
        np.ndarray: The original data.
        """
        return X * self.medians_
