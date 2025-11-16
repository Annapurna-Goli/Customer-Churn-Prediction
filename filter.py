""" filter methods """

import numpy as np
import sys
import logging
from sklearn.feature_selection import SelectKBest, f_classif
logger = logging.getLogger('filter')


class FeatureSelector:
    def __init__(self, X_train, X_test, y_train):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train


    def remove_duplicate_features(self):
        #Remove duplicate columns that have identical data.
        try:
            before_cols = len(self.X_train.columns)
            duplicated = self.X_train.T.duplicated()
            dup_cols = self.X_train.columns[duplicated].tolist()

            if dup_cols:
                self.X_train.drop(columns=dup_cols, inplace=True)
                self.X_test.drop(columns=dup_cols, inplace=True, errors="ignore")
                logger.info(f"Removed duplicate columns: {dup_cols}")
            else:
                logger.info("No duplicate columns found.")

            after_cols = len(self.X_train.columns)
            logger.info(f"Duplicate filter: {before_cols} → {after_cols} columns")
            logger.info(f"Columns now: {list(self.X_train.columns)}")

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Error in remove_duplicate_features: {er_msg} at {er_lin.tb_lineno}")
            raise e


    def apply_anova(self, top_k=25):
        """Apply ANOVA F-test but ensure no key features are lost."""
        try:
            before_cols = len(self.X_train.columns)
            numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()


            if len(numeric_cols) < 3:
                logger.info("Skipping ANOVA: not enough numeric columns.")
                return

            X_num = self.X_train[numeric_cols]
            selector = SelectKBest(score_func=f_classif, k=min(top_k, len(X_num.columns)))
            selector.fit(X_num, self.y_train)
            kept = list(X_num.columns[selector.get_support()])

            # Safety check: never drop key numeric columns
            must_keep = [
                'SeniorCitizen_yeojohnson',
                'tenure_yeojohnson',
                'MonthlyCharges_yeojohnson',
                'TotalCharges_knn_yeojohnson',
                'Contract_con',
                'sim'
            ]
            for col in must_keep:
                if col in self.X_train.columns and col not in kept:
                    kept.append(col)

            # final safe set
            self.X_train = self.X_train[kept]
            self.X_test = self.X_test[kept]

            after_cols = len(self.X_train.columns)
            logger.info(f"ANOVA retained {len(kept)} columns: {kept}")
            logger.info(f"ANOVA filter: {before_cols} → {after_cols} columns")
            logger.info(f"Columns now: {list(self.X_train.columns)}")

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Error in apply_anova: {er_msg} at {er_lin.tb_lineno}")
            raise e


    def filter_method(self):
        logger.info("Starting advanced feature filtering...")

        before_cols = len(self.X_train.columns)
        logger.info(f"Columns before filtering: {before_cols}")
        logger.info(f"Column names before filtering: {list(self.X_train.columns)}")

        # filters — no low variance or correlation drop
        self.remove_duplicate_features()
        self.apply_anova(top_k=25)

        after_cols = len(self.X_train.columns)
        logger.info(f"✅ Final Train shape after safe filtering: {self.X_train.shape}")
        logger.info(f"✅ Final Test shape after safe filtering: {self.X_test.shape}")
        logger.info(f"✅ Final columns retained ({after_cols}): {list(self.X_train.columns)}")

        logger.info(f"✅ Filtering complete — columns reduced from {before_cols} → {after_cols}")
        return self.X_train, self.X_test
