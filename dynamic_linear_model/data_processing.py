import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from config import config

class DataPreprocessing:
    def __init__(self, file_path: str, brand: str, dependent_variable: str, independent_variables_X: List[str], independent_variables_Z: List[str]):
        """
        Initialize the DataPreprocessing class with the necessary parameters.

        :param file_path: Path to the CSV file containing the data.
        :param brand: Brand name to filter the data.
        :param dependent_variable: The dependent variable column name.
        :param independent_variables_X: List of column names for the independent control variables X.
        :param independent_variables_Z: List of column names for the independent interested variables Z.
        """
        self.file_path = file_path
        self.brand = brand
        self.dependent_variable = dependent_variable
        self.independent_variables_X = independent_variables_X
        self.independent_variables_Z = independent_variables_Z
        self.scaler_X = AbsoluteMedianScaler()
        self.scaler_Z = AbsoluteMedianScaler()
        self.scaler_Y = AbsoluteMedianScaler()

    def preprocess(self, normalization: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Preprocess the data by loading, filtering, extracting variables, and optionally normalizing them.

        :param normalization: Boolean flag to indicate whether to normalize the data.
        :return: Tuple containing the normalized (or original) X_t, Z_t, and Y_t.
        """
        df = self._load_data()
        df = self._filter_brand(df)
        Y_t = self._dependent_variable(df)
        X_t, Z_t = self._independent_variable(df)
        if normalization:
            X_t, Z_t, Y_t = self._normalize_data(X_t, Z_t, Y_t)
        return X_t, Z_t, Y_t, 
    
    def get_normalized_scaler(self):
        return self.scaler_X, self.scaler_Z, self.scaler_Y
    
    def normalize_by_median(self, matrix):
        medians = np.median(matrix, axis=0)
        normalized_matrix = matrix - medians
        return normalized_matrix


    def _load_data(self) -> pd.DataFrame:
        """
        Load the data from a CSV file.

        :return: DataFrame containing the loaded data.
        """
        df = pd.read_csv(self.file_path)
        return df

    def _filter_brand(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the data by the specified brand.

        :param df: DataFrame containing the data.
        :return: Filtered DataFrame containing only the specified brand.
        """
        df = df[df["brand"] == self.brand]
        return df

    def _dependent_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract the dependent variable from the data.

        :param df: DataFrame containing the data.
        :return: Series containing the dependent variable values.
        """
        Y_t = df[self.dependent_variable].values
        return Y_t

    def _independent_variable(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract the independent variables X and Z from the data.

        :param df: DataFrame containing the data.
        :return: Tuple containing the DataFrames for independent variables X and Z.
        """
        X_t = df[self.independent_variables_X].values
        Z_t = df[self.independent_variables_Z].values
        return X_t, Z_t

    def _normalize_data(self, X_t: np.array, Z_t: np.array, Y_t: np.array) -> Tuple[np.array, np.array, np.array]:
        """
        Normalize the independent and dependent variables.

        :param X_t: DataFrame containing the independent variables X.
        :param Z_t: DataFrame containing the independent variables Z.
        :param Y_t: Series containing the dependent variable values.
        :return: Tuple containing the normalized X_t, Z_t, and Y_t.
        """
        if config["dataset"]["isNormalizeX"]:
            X_t_normalized = self.scaler_X.fit_transform(X_t)
        else:
            X_t_normalized = X_t
        Z_t_normalized = self.scaler_Z.fit_transform(Z_t)
        Y_t_normalized = self.scaler_Y.fit_transform(Y_t.reshape(-1, 1)).flatten()

        print("X_t_normalized", X_t_normalized)
        print("Z_t_normalized", Z_t_normalized)
        print("Y_t_normalized", Y_t_normalized)

        return X_t_normalized, Z_t_normalized, Y_t_normalized


class AbsoluteMedianScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Calculate the median of each feature
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
    
    def transform(self, X):
        # Scale the data by the absolute value of the median of each column
        return X / self.medians_
    
    def inverse_transform(self, X):
        # Scale back the data by multiplying with the median of each column
        return X * self.medians_