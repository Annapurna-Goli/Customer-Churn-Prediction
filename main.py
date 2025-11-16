"""
In this file we load data and perform all preprocessing steps for model development.
"""



import pandas as pd
import os
import sys
import logging
from sklearn.model_selection import train_test_split
from dupmissing import MissingValues
from log import setup_logging
import warnings
warnings.filterwarnings("ignore")
logger = setup_logging('main')
from variabletrans import VAR_TRANS
from outliers import OutlierHandler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from filter import FeatureSelector
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pickle
from model import ModelDevelopment





class TELICOM_CHR_INFO:
    def __init__(self, path):
        try:
            self.OUTPUT_DIR = r"C:\Users\golis\OneDrive\Desktop\crps\output"
            self.df = pd.read_csv(path)
            # Clean blank-like strings
            self.df = self.df.replace(['', ' ', 'Blank', 'NULL', 'NaN'], pd.NA)
            logger.info(f"Data loaded successfully: {self.df.shape}")
            # Drop unnecessary column
            self.df = self.df.drop(['customerID'], axis=1)
            logger.info(f"Missing values in dataset:\n{self.df.isnull().sum()}")

            self.X = self.df.iloc[:, :-1] #independent
            self.y = self.df.iloc[:, -1] # dependent


            # Split into train/test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            logger.info("Train-test split successful.")


        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in __init__: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e

    def _missing_val(self):
        try:
            # Create MissingValues object and perform KNN imputation
            mv_obj = MissingValues(self.X_train, self.X_test)
            self.X_train, self.X_test = mv_obj.knn_impute()

            # Drop duplicate column created after KNN imputation
            if 'TotalCharges' in self.X_train.columns:
                self.X_train = self.X_train.drop(['TotalCharges'], axis=1)
            if 'TotalCharges' in self.X_test.columns:
                self.X_test = self.X_test.drop(['TotalCharges'], axis=1)

            logger.info(f"Null values post-imputation (train):\n{self.X_train.isnull().sum()}")
            logger.info(f"Null values post-imputation (test):\n{self.X_test.isnull().sum()}")
            #logger.info(f"Null values post-imputation (test:\n{self.X_train.tolist()}")
            print(f'{self.X_train.info()}')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')

            logger.info(f"Cat Column names:\n{self.X_train_cat.columns.tolist()}")
            logger.info(f"Num Column names:\n{self.X_train_num.columns.tolist()}")
            logger.info(f"AFTER <misssing values>: Train num={self.X_train_num.shape[0]}, Test num={self.X_test_num.shape[0]}")
            # create '_orig' for plotting readability
            self.X_train_num_raw = self.X_train_num.copy()
            self.X_test_num_raw = self.X_test_num.copy()
            self.X_train_num_raw['tenure_orig'] = self.X_train_num_raw['tenure']
            self.X_train_num_raw['MonthlyCharges_orig'] = self.X_train_num_raw['MonthlyCharges']
            self.X_train_num_raw['TotalCharges_orig'] = self.X_train_num_raw['TotalCharges_knn']

            self.X_test_num_raw['tenure_orig'] = self.X_test_num_raw['tenure']
            self.X_test_num_raw['MonthlyCharges_orig'] = self.X_test_num_raw['MonthlyCharges']
            self.X_test_num_raw['TotalCharges_orig'] = self.X_test_num_raw['TotalCharges_knn']




        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in __init__: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e

    def var_transform(self):
        try:
            #VAR_TRANS(self.X_train_num,self.X_train_num)
            vart_obj = VAR_TRANS(self.X_train_num, self.X_test_num)
            self.X_train_num, self.X_test_num = vart_obj.yeojohnson_transform()
            vart_obj.visualize_all_transformations()
            logger.info(f"AFTER <variable transformation>: Train num={self.X_train_num.shape[0]}, Test num={self.X_test_num.shape[0]}")

            # keeping copy for visualization later. Keep raw copies for readability
            self.X_train_num_raw = self.X_train_num.copy()
            self.X_test_num_raw = self.X_test_num.copy()



        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in __init__: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e

    def outlier_method(self):
        try:
            # Assuming self.X_train_num and self.X_test_num exist and are the numeric subsets
            numeric_cols = self.X_train_num.select_dtypes(include=['int64', 'float64']).columns

            # Initialize the handler
            vart_obj = OutlierHandler(self.X_train_num, self.X_test_num, numeric_cols)

            for col in numeric_cols:
                vart_obj.gaussian_outliers(col)
                vart_obj.iqr_outliers(col)

            vart_obj.isolation_forest_visualize_pca()

            # 1. Run capping and capture the returned (capped) DataFrames
            train_capped_df, test_capped_df = vart_obj.cap_outliers()

            # 2. Update your main class variables with the new, capped data
            self.X_train_num = train_capped_df

            self.X_test_num = test_capped_df

            # 3. Pass the capped data to the visualization function
            vart_obj.visualize_post_capping(train_capped_df, test_capped_df)
            logger.info(f"AFTER <outlier handling>: Train num={self.X_train_num.shape[0]}, Test num={self.X_test_num.shape[0]}")

            # keeping original col so that i can use later for visualization
            #self.X_train_num_raw = self.X_train_num.copy()
            #self.X_test_num_raw = self.X_test_num.copy()



        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in __init__: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e


    def encoding_method(self):
        try:
            print(f"train :{self.X_train_cat.columns.tolist()}")
            # Log the full categorical DataFrame
            logger.info(f"\n{self.X_train_cat.head(10)}")
            logger.info(f"training num dataaaaa:::\n{self.X_train_num.head(10)}")
            print(f" train :\n{self.X_train_cat.head(10)}")

            logger.info(f"test : {self.X_test_cat.columns.tolist()}")
            print(f"test : {self.X_test_cat.columns.tolist()}")
            logger.info(f"test num dataaaaa:::\n{self.X_test_num.head(10)}")
            logger.info(f"yyyyyy ... training num dataaaaa:::\n{self.y_train.head(10)}")
            # Log the full categorical DataFrame
            logger.info(f" test : \n{self.X_test_cat.head(10)}")
            print(f" test : \n{self.X_test_cat.head(10)}")
            # applying onehot encoding
            oh = OneHotEncoder(categories='auto', drop='first',
                               handle_unknown='ignore')
            oh.fit(self.X_train_cat[[
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                 'PaperlessBilling', 'PaymentMethod','gateway', 'tax']])
            logger.info(f'{oh.categories_}')
            print(f'{oh.categories_}')
            logger.info(f'{oh.get_feature_names_out()}')
            print(f'{oh.get_feature_names_out()}')
            res = oh.transform(self.X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'PaperlessBilling', 'PaymentMethod',  'gateway', 'tax']]).toarray()
            res_test = oh.transform(self.X_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                 'PaperlessBilling', 'PaymentMethod', 'gateway', 'tax']]).toarray()
            f = pd.DataFrame(res, columns=oh.get_feature_names_out())
            f_test = pd.DataFrame(res_test, columns=oh.get_feature_names_out())
            self.X_train_cat.reset_index(drop=True, inplace=True)
            f.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)
            f_test.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, f], axis=1)
            self.X_test_cat = pd.concat([self.X_test_cat, f_test], axis=1)
            #self.X_train_cat['SeniorCitizen_yeojohnson'] = self.X_train_num['SeniorCitizen_yeojohnson']
            #self.X_test_cat['SeniorCitizen_yeojohnson'] = self.X_test_num['SeniorCitizen_yeojohnson']
            self.X_train_cat['sim'] = self.X_train_cat['sim'].map({'Jio': 0, 'Airtel': 1, 'Vi': 2, 'BSNL': 3})
            self.X_test_cat['sim'] = self.X_test_cat['sim'].map({'Jio': 0, 'Airtel': 1, 'Vi': 2, 'BSNL': 3})
            logger.info(self.X_test_cat['sim'])
            #self.X_train_num = self.X_train_num.drop(['SeniorCitizen_yeojohnson'], axis=1)
            #self.X_test_num = self.X_test_num.drop(['SeniorCitizen_yeojohnson'], axis=1)

            # rented_Contract we are going apply Odinal Encoder

            od = OrdinalEncoder()
            od.fit(self.X_train_cat[['Contract']])
            logger.info(f'{od.categories_}')
            logger.info(f' column_names: {od.get_feature_names_out()}')
            res1 = od.transform(self.X_train_cat[['Contract']])
            res1_test = od.transform(self.X_test_cat[['Contract']])
            c_names = od.get_feature_names_out()
            f1 = pd.DataFrame(res1, columns=c_names + ['_con'])
            f1_test = pd.DataFrame(res1_test, columns=c_names + ['_con'])
            self.X_train_cat.reset_index(drop=True, inplace=True)
            f1.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)
            f1_test.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, f1], axis=1)
            self.X_test_cat = pd.concat([self.X_test_cat, f1_test], axis=1)
            logger.info(f'train null {self.X_train_cat.isnull().sum()}')
            self.X_train_cat = self.X_train_cat.drop(
                ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'PaperlessBilling', 'PaymentMethod', 'gateway', 'tax','Contract'], axis=1)
            self.X_test_cat = self.X_test_cat.drop(
                ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'PaperlessBilling', 'PaymentMethod',  'gateway', 'tax','Contract'],
                axis=1)
            logger.info(f'train null {self.X_train_cat.isnull().sum()}')
            logger.info(f'test null {self.X_test_cat.isnull().sum()}')

            logger.info(f'{self.X_train_cat.sample(10)}')
            logger.info(f'{self.X_test_cat.sample(10)}')

            logger.info(f'y_train_data : {self.y_train.unique()}')
            logger.info(f'y_train_data : {self.y_train.isnull().sum()}')
            logger.info(f'y_test_data : {self.y_test.unique()}')
            logger.info(f'y_test_data : {self.y_test.isnull().sum()}')

            # dependent varibale should be converted using label encoder
            logger.info(f'{self.y_train[:10]}')
            lb = LabelEncoder()
            lb.fit(self.y_train)
            self.y_train = lb.transform(self.y_train)
            self.y_test = lb.transform(self.y_test)
            logger.info(f'detailed : {lb.classes_} ')
            logger.info(f'{self.y_train[:10]}')
            logger.info(f'y_train_data : {self.y_train.shape}')
            logger.info(f'y_test_data : {self.y_test.shape}')
            logger.info(f'Check-label{self.y_train}')

            # check the shape to concate for next filtering method

            logger.info(f":shape to concate for next filtering method:\n{self.X_train_num.shape}")
            logger.info(f":shape to concate for next filtering method:\n{self.X_train_cat.shape}")
            logger.info(f":shape to concate for next filtering method:\n{self.X_test_num.shape}")
            logger.info(f":shape to concate for next filtering method:\n{self.X_test_cat.shape}")

            logger.info(f"Before merge:")
            logger.info(f"Train num shape: {self.X_train_num.shape}")
            logger.info(f"Train cat shape: {self.X_train_cat.shape}")
            logger.info(f"Test num shape: {self.X_test_num.shape}")
            logger.info(f"Test cat shape: {self.X_test_cat.shape}")

            # Identify mismatch rows
            if len(self.X_test_num) != len(self.X_test_cat):
                logger.error(f" Row mismatch in TEST: num={len(self.X_test_num)}, cat={len(self.X_test_cat)}")
                diff = abs(len(self.X_test_num) - len(self.X_test_cat))
                logger.error(f"Difference of {diff} rows!")


            self.X_train_num = self.X_train_num.reset_index(drop=True)
            self.X_test_num = self.X_test_num.reset_index(drop=True)
            self.X_train_cat = self.X_train_cat.reset_index(drop=True)
            self.X_test_cat = self.X_test_cat.reset_index(drop=True)
            logger.info(f'rows to check the categorical values 10 rows after encoding:{self.X_train_cat.iloc[:10]}')

            # sanity check before merge
            assert len(self.X_train_num) == len(self.X_train_cat), "Train rows mismatch!"
            assert len(self.X_test_num) == len(self.X_test_cat), "Test rows mismatch!"

            # merging
            self.X_train_final = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.X_test_final = pd.concat([self.X_test_num, self.X_test_cat], axis=1)

            logger.info(f" Merged final train shape: {self.X_train_final.shape}")
            logger.info(f"Merged final test shape: {self.X_test_final.shape}")
            logger.info(f"to check senior citizenship name sample values:\n{self.X_train_final.head()}")


        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in __init__: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e

    def _filter_features(self):
        try:
            logger.info("Starting advanced feature filtering...")

            filt_obj = FeatureSelector(self.X_train_final, self.X_test_final, self.y_train)
            self.X_train_final, self.X_test_final = filt_obj.filter_method()

            logger.info(f" Final Train shape after safe filtering: {self.X_train_final.shape}")
            logger.info(f" Final Test shape after safe filtering: {self.X_test_final.shape}")
            logger.info(f" Final columns retained:\n{self.X_train_final.columns.tolist()}")

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Error in _filter_features: {er_msg} at line {er_lin.tb_lineno}")
            raise

    def balancing(self):
        try:
            logger.info('----------------Before Balancing------------------------')
            logger.info(f'Total row for yes category in training data {self.X_train_final.shape[0]} was : {sum(self.y_train == 1)}')
            logger.info(f"✅ shape for test shape: {self.X_train_final.shape}")
            logger.info(f'Total row for no category in training data {self.X_train_final.shape[0]} was : {sum(self.y_train == 0)}')
            logger.info(f'---------------After Balancing-------------------------')
            sm = SMOTE(random_state=42)
            self.X_train_final_res, self.y_train_res = sm.fit_resample(self.X_train_final, self.y_train)
            logger.info(f'Total row for yes category in training data {self.X_train_final_res.shape[0]} was : {sum(self.y_train_res == 1)}')
            logger.info(f"after balancing shape final train shape: {self.X_train_final_res.shape}")
            logger.info(f"after balancing shape final for y train  shape: {self.y_train_res.shape}")
            logger.info(f'Total row for no category in training data {self.X_train_final.shape[0]} was : {sum(self.y_train_res == 0)}')
            #to check data final data set
            final_train = self.X_train_final_res.copy()  # balanced features
            final_train['target'] = self.y_train_res
            print(final_train.head())
            print(final_train.shape)

            final_test = self.X_test_final.copy()
            final_test['target'] = self.y_test
            print(f'test data final : final_test.head()')
            print(f'test data shape for final :final_test.shape')
            # Check shapes after SMOTE
            logger.info(
                f"Post-SMOTE shapes — X_train_final_res: {self.X_train_final_res.shape}, y_train_res: {self.y_train_res.shape}")

            if len(self.X_train_final_res) != len(self.y_train_res):
                logger.error(" Mismatch in X_train_final_res and y_train_res rows after SMOTE!")
            else:
                logger.info(" Shapes match after SMOTE.")


            print("MonthlyCharges min/max:", self.X_train_final_res['MonthlyCharges_yeojohnson'].min(), self.X_train_final_res['MonthlyCharges_yeojohnson'].max())
            print("TotalCharges min/max:", self.X_train_final_res['TotalCharges_knn_yeojohnson'].min(), self.X_train_final_res['TotalCharges_knn_yeojohnson'].max())




        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def scale_numeric_features_post_smote(self):
        try:
            logger.info("Starting numeric scaling on post-SMOTE data...")

            # SDefine numeric columns to scale (only those in the dataset)
            numeric_cols = ['MonthlyCharges_yeojohnson', 'TotalCharges_knn_yeojohnson']
            numeric_cols = [col for col in numeric_cols if col in self.X_train_final_res.columns]

            if not numeric_cols:
                logger.warning("No numeric columns found for scaling.")
                return

            #  Backup non-numeric columns to merge later
            other_cols_train = self.X_train_final_res.drop(columns=numeric_cols, errors='ignore')
            other_cols_test = self.X_test_final.drop(columns=numeric_cols, errors='ignore')

            # Fit scaler on post-SMOTE training numeric data
            self.scaler_post_smote = StandardScaler()
            self.scaler_post_smote.fit(self.X_train_final_res[numeric_cols])

            # Transform numeric columns
            scaled_train_df = pd.DataFrame(
                self.scaler_post_smote.transform(self.X_train_final_res[numeric_cols]),
                columns=numeric_cols,
                index=self.X_train_final_res.index
            )
            scaled_test_df = pd.DataFrame(
                self.scaler_post_smote.transform(self.X_test_final[numeric_cols]),
                columns=numeric_cols,
                index=self.X_test_final.index
            )

            # Concatenate scaled numeric columns back with the rest
            self.X_train_final_res_scaled = pd.concat(
                [other_cols_train.reset_index(drop=True), scaled_train_df.reset_index(drop=True)], axis=1
            )
            self.X_test_final_scaled = pd.concat(
                [other_cols_test.reset_index(drop=True), scaled_test_df.reset_index(drop=True)], axis=1
            )

            # Replace old variables with scaled versions
            self.X_train_final_res = self.X_train_final_res_scaled.copy()
            self.X_test_final = self.X_test_final_scaled.copy()

            #  Save scaler for future use
            with open(os.path.join(self.OUTPUT_DIR, "scaler_post_smote.pkl"), "wb") as f:
                pickle.dump(self.scaler_post_smote, f)

            logger.info(f"Numeric columns scaled and merged successfully: {numeric_cols}")
            logger.info(
                f"Scaled train shape: {self.X_train_final_res.shape}, Scaled test shape: {self.X_test_final.shape}")
            print("FINAL TRAINING COLUMNS:", list(obj.X_train_final_res.columns))


        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in main: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e

    def model_development_main(self):

        try:
            MODEL_COLUMNS = [
                'SeniorCitizen_yeojohnson', 'tenure_yeojohnson', 'sim', 'gender_Male', 'Partner_Yes',
                'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
                'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
                'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes',
                'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                'PaymentMethod_Mailed check', 'gateway_Enabled', 'Contract_con', 'MonthlyCharges_yeojohnson',
                'TotalCharges_knn_yeojohnson'
            ]


            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)

            print("Columns and first row values before model prediction:")
            print(self.X_train_final.iloc[:10])  # <- use your actual final data variable here

            # Reset display options if needed
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')
            logger.info("Starting final model development")
            # checks before model training
            logger.info(f"X_train_final shape: {self.X_train_final.shape}")
            logger.info(f"y_train_res shape: {self.y_train_res.shape}")
            logger.info(f"X_test_final shape: {self.X_test_final.shape}")
            logger.info(f"y_test shape: {self.y_test.shape}")
            print(f'list with values before sending to model after scaling which is last step :{self.X_train_final.head()}')
            logger.info(
                f'list with values before sending to model after scaling which is last step :{self.X_train_final.head()}')
            print("Columns and first row values before model prediction:")
            print(self.X_train_final.iloc[:10])
            logger.info(f'Columns and first row values before model prediction:{self.X_train_final.iloc[:10]}')

            extra_cols = [c for c in self.X_train_final_res.columns if c not in MODEL_COLUMNS]
            print("Extra columns in the input row before model prediction:", extra_cols)
            logger.info(f"Extra columns in the input row before model prediction: {extra_cols}")
            self.X_train_final_res = self.X_train_final_res[MODEL_COLUMNS]


            # Check shapes
            print("Shape checks before model training:")
            print("X_train_final:", self.X_train_final.shape)
            print("y_train_res:", self.y_train_res.shape)
            print("X_test_final:", self.X_test_final.shape)
            print("y_test:", self.y_test.shape)
            # Initialize model class
            model_dev = ModelDevelopment()
            best_model = model_dev.common(
                self.X_train_final_res,  # balanced training features
                self.y_train_res,  # balanced target
                self.X_test_final,  # test features stay the same
                self.y_test  # test target
            )

            logger.info(f"X_train_final_res: Column names:\n{self.X_train_final_res.columns.tolist()}")

            logger.info(f"X_test_final: Column names:\n{self.X_test_final.columns.tolist()}")

            if best_model is not None:
                logger.info("Model development completed successfully.")
            else:
                logger.error("Model training failed, no model returned.")

            return best_model


        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue in main: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
            raise e


if __name__ == "__main__":
    try:

        path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
        obj = TELICOM_CHR_INFO(path)
        obj._missing_val()
        obj.var_transform()
        obj.outlier_method()
        obj.encoding_method()
        obj._filter_features()
        obj.balancing()
        obj.scale_numeric_features_post_smote()
        obj. model_development_main()


    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.error(f"Issue in main: {er_ty}: {er_msg} at line {er_lin.tb_lineno}")
        raise e
