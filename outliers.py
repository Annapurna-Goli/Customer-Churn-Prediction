import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
logger = logging.getLogger('outliers')



class OutlierHandler:

    # Define col to skip capping/visualization once for consistency
    FEATURES_TO_SKIP_CAPPING = {'SeniorCitizen_yeojohnson'}

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: list):
        """Initializes the OutlierHandler with copies of the dataframes."""
        # This will be the capped data after running cap_outliers
        self.train_num = train_df.copy()
        self.test_num = test_df.copy()

        # Store the originals for comparison in visualization
        self.train_num_original = train_df.copy()
        self.test_num_original = test_df.copy()

        self.numeric_cols = numeric_cols

    # Gaussian Outlier Detection (Train + Test)

    def gaussian_outliers(self, col: str):

        try:
            mean_train = self.train_num[col].mean()
            std_train = self.train_num[col].std()
            lower, upper = mean_train - 3 * std_train, mean_train + 3 * std_train

            outliers_train = self.train_num[(self.train_num[col] < lower) | (self.train_num[col] > upper)]
            outliers_test = self.test_num[(self.test_num[col] < lower) | (self.test_num[col] > upper)]

            logger.info(f" {col}: Train Gaussian outliers = {len(outliers_train)}, Test = {len(outliers_test)}")

            # Visualization
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(self.train_num[col], kde=True, color='blue')
            plt.axvline(lower, color='red', linestyle='--', label='-3σ')
            plt.axvline(upper, color='red', linestyle='--', label='+3σ')
            plt.title(f"Train - Gaussian Outliers ({col})")
            plt.legend()

            plt.subplot(1, 2, 2)
            sns.histplot(self.test_num[col], kde=True, color='green')
            plt.axvline(lower, color='red', linestyle='--')
            plt.axvline(upper, color='red', linestyle='--')
            plt.title(f"Test - Gaussian Outliers ({col})")

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)  # show for 1 second
            plt.close()
            return outliers_train, outliers_test
        except Exception as e:
            er_ty, er_msg, er_tb = sys.exc_info()
            logger.error(f"Issue in gaussian_outliers: {er_ty.__name__} - {er_msg} at line {er_tb.tb_lineno}")

    # IQR Outlier Detection (Train + Test)

    def iqr_outliers(self, col: str):

        try:
            Q1 = self.train_num[col].quantile(0.25)
            Q3 = self.train_num[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

            outliers_train = self.train_num[(self.train_num[col] < lower) | (self.train_num[col] > upper)]
            outliers_test = self.test_num[(self.test_num[col] < lower) | (self.test_num[col] > upper)]

            logger.info(f" {col}: Train IQR outliers = {len(outliers_train)}, Test = {len(outliers_test)}")

            # Visualization
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            sns.boxplot(x=self.train_num[col])
            plt.title(f"Train - IQR Outliers ({col})")

            plt.subplot(1, 2, 2)
            sns.boxplot(x=self.test_num[col])
            plt.title(f"Test - IQR Outliers ({col})")

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)  # show for 1 second
            plt.close()
            return outliers_train, outliers_test
        except Exception as e:
            er_ty, er_msg, er_tb = sys.exc_info()
            logger.error(f"Issue in iqr_outliers: {er_ty.__name__} - {er_msg} at line {er_tb.tb_lineno}")

    # Isolation Forest Detection (Train)

    def isolation_forest_visualize_pca(self):

        try:
            # Fit Isolation Forest on train data (Contamination set to 5%)
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds_train = iso.fit_predict(self.train_num)

            # Identify outliers (-1 indicates an anomaly)
            train_outliers = self.train_num[preds_train == -1]
            logger.info(f"Isolation Forest: Train outliers = {len(train_outliers)}")


            pca = PCA(n_components=2)
            components = pca.fit_transform(self.train_num)

            plt.figure(figsize=(8, 6))
            # Use coolwarm colormap: anomalies (-1) are one color, normal (1) is another
            plt.scatter(components[:, 0], components[:, 1], c=preds_train, cmap='coolwarm', s=20)
            plt.title("Isolation Forest Outliers (PCA Visualization)")
            plt.xlabel("Principal Component 1 (PC1)")
            plt.ylabel("Principal Component 2 (PC2)")
            plt.show(block=False)
            plt.pause(1)
            plt.close()

        except Exception as e:
            er_ty, er_msg, er_tb = sys.exc_info()
            logger.error(f" Issue in isolation_forest_visualize_pca: {er_ty.__name__} - {er_msg} at line {er_tb.tb_lineno}")

    #  Apply 5th–95th Quantile Capping (Train + Test)

    def cap_outliers(self) -> tuple[pd.DataFrame, pd.DataFrame]:

        #Applies 5th and 95th percentile capping to non-excluded numeric columns.

        try:
            for col in self.numeric_cols:
                if col in self.FEATURES_TO_SKIP_CAPPING:
                    logger.info(f"{col}: Skipping capping (Derived from Binary feature).")
                    continue

                # Quantiles based only on the TRAINING data
                lower = self.train_num[col].quantile(0.05)
                upper = self.train_num[col].quantile(0.95)

                # Apply clipping to both Train and Test sets
                self.train_num[col] = np.clip(self.train_num[col], lower, upper)
                self.test_num[col] = np.clip(self.test_num[col], lower, upper)

                logger.info(f" {col}: Capped [Train/Test] between 5th ({lower:.3f}) and 95th ({upper:.3f}) percentiles")

            return self.train_num, self.test_num

        except Exception as e:
            er_ty, er_msg, er_tb = sys.exc_info()
            logger.error(f"Issue in cap_outliers: {er_ty.__name__} - {er_msg} at line {er_tb.tb_lineno}")

    # Visualize Before & After Capping (Train + Test)

    def visualize_post_capping(self, capped_train: pd.DataFrame, capped_test: pd.DataFrame):
        #Visualizes the effect of capping using boxplots.
        try:
            for col in self.numeric_cols:
                if col in self.FEATURES_TO_SKIP_CAPPING:
                    logger.info(f" {col}: Skipping visualization (Was skipped during capping).")
                    continue

                # Plotting logic runs ONLY if the column was not skipped
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 2, 1)
                sns.boxplot(x=self.train_num_original[col])
                plt.title(f"Train - Before Capping ({col})")

                plt.subplot(2, 2, 2)
                sns.boxplot(x=capped_train[col])
                plt.title(f"Train - After Capping ({col})")

                plt.subplot(2, 2, 3)
                sns.boxplot(x=self.test_num_original[col])
                plt.title(f"Test - Before Capping ({col})")

                plt.subplot(2, 2, 4)
                sns.boxplot(x=capped_test[col])
                plt.title(f"Test - After Capping ({col})")

                plt.tight_layout()
                plt.show(block=False)
                plt.pause(1)
                plt.close()

        except Exception as e:
            er_ty, er_msg, er_tb = sys.exc_info()
            logger.error(f" Issue in visualize_post_capping: {er_ty.__name__} - {er_msg} at line {er_tb.tb_lineno}")