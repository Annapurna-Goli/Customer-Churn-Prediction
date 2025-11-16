import numpy as np
import pandas as pd
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox
from scipy.stats import yeojohnson
from sklearn.preprocessing import PowerTransformer
from log import setup_logging
import warnings
warnings.filterwarnings("ignore")
logger = setup_logging('variabletrans')


class VAR_TRANS:
    def __init__(self, x_train_num, x_test_num):
        try:
            self.x_train_num = x_train_num.copy()
            self.x_test_num = x_test_num.copy()
            self.base_cols = x_train_num.columns.tolist()   # ✅ Define base columns once


        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in __init__: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e

    def _plot_distribution(self, df, var, title_prefix):
        """Plot distribution, boxplot, and probplot for a given variable."""
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title(f'{title_prefix} - KDE')
        df[var].plot(kind='kde', color='r')

        plt.subplot(1, 3, 2)
        plt.title(f'{title_prefix} - Boxplot')
        sns.boxplot(x=df[var])

        plt.subplot(1, 3, 3)
        plt.title(f'{title_prefix} - Probplot')
        stats.probplot(df[var], dist='norm', plot=plt)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    def _safe_transform(self, series, func, desc):
        try:
            transformed = func(series)
            logger.info(f"{desc} applied successfully on {series.name}")
            return transformed
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in {desc}: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e

    def log_transform(self):
        for i in self.base_cols:
            self.x_train_num[i + '_log'] = np.log1p(self.x_train_num[i])
            self.x_test_num[i + '_log'] = np.log1p(self.x_test_num[i])
            self._plot_distribution(self.x_train_num, i + '_log', 'Log Transform')
        logger.info("Log transformation completed successfully.")
        return self.x_train_num, self.x_test_num

    def reciprocal_transform(self):
        for i in self.base_cols:
            self.x_train_num[i + '_reciprocal'] = 1 / (self.x_train_num[i] + 1e-6)
            self.x_test_num[i + '_reciprocal'] = 1 / (self.x_test_num[i] + 1e-6)
            self._plot_distribution(self.x_train_num, i + '_reciprocal', 'Reciprocal Transform')
        logger.info("Reciprocal transformation completed successfully.")
        return self.x_train_num, self.x_test_num

    def sqrt_transform(self):
        for i in self.base_cols:
            self.x_train_num[i + '_sqrt'] = np.sqrt(np.abs(self.x_train_num[i]))
            self.x_test_num[i + '_sqrt'] = np.sqrt(np.abs(self.x_test_num[i]))
            self._plot_distribution(self.x_train_num, i + '_sqrt', 'Square Root Transform')
        logger.info("Square root transformation completed successfully.")
        return self.x_train_num, self.x_test_num

    def exp_transform(self):
        for i in self.base_cols:
            self.x_train_num[i + '_exp'] = np.exp(self.x_train_num[i] / self.x_train_num[i].max())
            self.x_test_num[i + '_exp'] = np.exp(self.x_test_num[i] / self.x_test_num[i].max())
            self._plot_distribution(self.x_train_num, i + '_exp', 'Exponential Transform')
        logger.info("Exponential transformation completed successfully.")
        return self.x_train_num, self.x_test_num

    def boxcox_transform(self):
        for i in self.base_cols:
            data = self.x_train_num[i]
            if (data <= 0).any():
                data = data - data.min() + 1
            transformed, _ = stats.boxcox(data)
            self.x_train_num[i + '_boxcox'] = transformed

            data_test = self.x_test_num[i]
            if (data_test <= 0).any():
                data_test = data_test - data_test.min() + 1
            self.x_test_num[i + '_boxcox'], _ = stats.boxcox(data_test)

            self._plot_distribution(self.x_train_num, i + '_boxcox', 'BoxCox Transform')
        logger.info("BoxCox transformation completed successfully.")
        return self.x_train_num, self.x_test_num

    def yeojohnson_transform(self):
        pt = PowerTransformer(method='yeo-johnson')
        for i in self.base_cols:
            reshaped_train = self.x_train_num[[i]].values
            reshaped_test = self.x_test_num[[i]].values
            self.x_train_num[i + '_yeojohnson'] = pt.fit_transform(reshaped_train)
            self.x_test_num[i + '_yeojohnson'] = pt.transform(reshaped_test)
            self._plot_distribution(self.x_train_num, i + '_yeojohnson', 'Yeo-Johnson Transform')

        # Keep only transformed columns
        f = [j for j in self.x_train_num.columns if '_yeojohnson' in j]
        self.x_train_num = self.x_train_num[f]
        self.x_test_num = self.x_test_num[[j for j in self.x_test_num.columns if j in f]]



        logger.info(f"Final columns after Yeo-Johnson transformation: {self.x_train_num.columns.tolist()}")
        return self.x_train_num, self.x_test_num

    def visualize_all_transformations(self):
        try:
            transformations = {
                'Log Transform': self.log_transform,
                'Reciprocal Transform': self.reciprocal_transform,
                'Square Root Transform': self.sqrt_transform,
                'Exponential Transform': self.exp_transform,
                'BoxCox Transform': self.boxcox_transform,
                'Yeo-Johnson Transform': self.yeojohnson_transform
            }

            for col in self.x_train_num.columns:  # only the transformed columns exist
                plt.figure()
                sns.histplot(self.x_train_num[col], kde=True)
                plt.title(f"{col} Distribution After Yeo-Johnson")
                plt.show(block=False)
                plt.pause(1)
                plt.close()

            logger.info("All transformation visualizations completed successfully.")

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in visualize_all_transformations: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e
