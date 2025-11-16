"""
handling missing values
"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
import matplotlib.pyplot as plt
import sys
import os
from log import setup_logging
logger = setup_logging('dupmissing')


class MissingValues:
    def __init__(self, x_train, x_test):
        try:
            self.x_train = x_train.copy()
            self.x_test = x_test.copy()
            self.f = []
            self.imputed_columns = {}

            for col in self.x_train.columns:
                if self.x_train[col].isnull().sum() > 0:
                    if self.x_train[col].dtype == object:
                        self.x_train[col] = pd.to_numeric(self.x_train[col], errors='coerce')
                        self.x_test[col] = pd.to_numeric(self.x_test[col], errors='coerce')
                    self.f.append(col)

            logger.info(f"Columns with missing values: {self.f}")

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in __init__: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e

    # ---------- Helper for STD plotting ----------
    def _plot_std_comparison(self, col, imputed_name):
        try:
            std_original = self.x_train[col].std()
            std_imputed = self.x_train[imputed_name].std()

            plt.figure(figsize=(4, 3))
            plt.bar(['Original', imputed_name], [std_original, std_imputed], color=['red', 'green'])
            plt.title(f"STD Comparison for {col} ({imputed_name})")
            plt.ylabel("Standard Deviation")
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            logger.info(f"STD for {col}: Original={round(std_original,3)}, {imputed_name}={round(std_imputed,3)}")

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in STD plot for {col}: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")

    # ---------- Mean Imputation ----------
    def mean_impute(self):
        for col in self.f:
            col_name = f"{col}_mean"
            self.x_train[col_name] = self.x_train[col].fillna(self.x_train[col].mean())
            self.x_test[col_name] = self.x_test[col].fillna(self.x_train[col].mean())
            self.imputed_columns.setdefault(col, []).append(col_name)
            self._plot_std_comparison(col, col_name)

    # ---------- Median Imputation ----------
    def median_impute(self):
        for col in self.f:
            col_name = f"{col}_median"
            self.x_train[col_name] = self.x_train[col].fillna(self.x_train[col].median())
            self.x_test[col_name] = self.x_test[col].fillna(self.x_train[col].median())
            self.imputed_columns.setdefault(col, []).append(col_name)
            self._plot_std_comparison(col, col_name)

    # ---------- Mode Imputation ----------
    def mode_impute(self):
        for col in self.f:
            col_name = f"{col}_mode"
            mode_val = self.x_train[col].mode()[0]
            self.x_train[col_name] = self.x_train[col].fillna(mode_val)
            self.x_test[col_name] = self.x_test[col].fillna(mode_val)
            self.imputed_columns.setdefault(col, []).append(col_name)
            self._plot_std_comparison(col, col_name)

    # ---------- Arbitrary Value Imputation ----------
    def arbitrary_impute(self, value=999):
        for col in self.f:
            col_name = f"{col}_arbitrary"
            self.x_train[col_name] = self.x_train[col].fillna(value)
            self.x_test[col_name] = self.x_test[col].fillna(value)
            self.imputed_columns.setdefault(col, []).append(col_name)
            self._plot_std_comparison(col, col_name)

    # ---------- Constant Imputation ----------
    def constant_impute(self, value=0):
        for col in self.f:
            col_name = f"{col}_constant"
            self.x_train[col_name] = self.x_train[col].fillna(value)
            self.x_test[col_name] = self.x_test[col].fillna(value)
            self.imputed_columns.setdefault(col, []).append(col_name)
            self._plot_std_comparison(col, col_name)

    # ---------- End of Distribution Imputation ----------
    def end_of_distribution_impute(self, factor=3):
        for col in self.f:
            col_name = f"{col}_end_dist"
            fill_val = self.x_train[col].mean() + factor * self.x_train[col].std()
            self.x_train[col_name] = self.x_train[col].fillna(fill_val)
            self.x_test[col_name] = self.x_test[col].fillna(fill_val)
            self.imputed_columns.setdefault(col, []).append(col_name)
            self._plot_std_comparison(col, col_name)

    # ---------- Random Sample Imputation ----------
    def random_sample_impute(self):
        for col in self.f:
            col_name = f"{col}_random"
            self.x_train[col_name] = self.x_train[col]
            self.x_test[col_name] = self.x_test[col]
            null_idx_train = self.x_train[col].isnull()
            null_idx_test = self.x_test[col].isnull()

            if null_idx_train.any():
                self.x_train.loc[null_idx_train, col_name] = np.random.choice(
                    self.x_train[col].dropna(), size=null_idx_train.sum()
                )
            if null_idx_test.any():
                self.x_test.loc[null_idx_test, col_name] = np.random.choice(
                    self.x_train[col].dropna(), size=null_idx_test.sum()
                )
            self.imputed_columns.setdefault(col, []).append(col_name)
            self._plot_std_comparison(col, col_name)

    # ---------- KNN Imputation ----------
    def knn_impute(self, n_neighbors=5):
        try:
            for col in self.f:
                col_name = f"{col}_knn"
                imputer = KNNImputer(n_neighbors=n_neighbors)
                self.x_train[col_name] = imputer.fit_transform(self.x_train[[col]])
                self.x_test[col_name] = imputer.transform(self.x_test[[col]])
                self.imputed_columns.setdefault(col, []).append(col_name)
                self._plot_std_comparison(col, col_name)

            logger.info("KNN imputation completed successfully.")
            return self.x_train, self.x_test

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in knn_impute: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e

    # ---------- Iterative Imputation ----------
    def iterative_impute(self):
        try:
            for col in self.f:
                col_name = f"{col}_iterative"
                imputer = IterativeImputer(random_state=42)
                self.x_train[col_name] = imputer.fit_transform(self.x_train[[col]])
                self.x_test[col_name] = imputer.transform(self.x_test[[col]])
                self.imputed_columns.setdefault(col, []).append(col_name)
                self._plot_std_comparison(col, col_name)

            logger.info("Iterative imputation completed successfully.")

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in iterative_impute: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e
