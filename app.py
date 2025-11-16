import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import logging

app = Flask(__name__)
app.secret_key = "replace-this-with-a-secure-key"

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_app")

# Load model and scaler
MODEL_PATH = os.path.join("output", "model.pkl")
SCALER_PATH = os.path.join("output", "scaler_post_smote.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Model expected columns
MODEL_COLUMNS = [
    'SeniorCitizen_yeojohnson', 'tenure_yeojohnson', 'sim', 'gender_Male', 'Partner_Yes', 'Dependents_Yes',
    'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic',
    'InternetService_No', 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes', 'TechSupport_Yes',
    'StreamingTV_Yes', 'StreamingMovies_Yes', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'gateway_Enabled', 'Contract_con',
    'MonthlyCharges_yeojohnson', 'TotalCharges_knn_yeojohnson'
]

# Mapping for categorical columns
YES_NO_MAP = {"Yes": 1, "No": 0}
GENDER_MAP = {"Male": 1, "Female": 0}

# Correct SIM mappings
SIM_FORM_TO_MODEL = {"0": 0, "1": 1, "2": 2, "3": 3}  # numeric value for model
SIM_DISPLAY_MAP = {0: "Jio", 1: "Airtel", 2: "Vi", 3: "BSNL"}  # text for display

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            user_input = request.form.to_dict()
            logger.info(f"Raw user input: {user_input}")

            # Numeric columns
            numeric_df = pd.DataFrame({
                'SeniorCitizen_yeojohnson': [float(user_input.get('SeniorCitizen', 0))],
                'tenure_yeojohnson': [float(user_input.get('tenure', 0))],
                'MonthlyCharges_yeojohnson': [float(user_input.get('MonthlyCharges', 0))],
                'TotalCharges_knn_yeojohnson': [float(user_input.get('TotalCharges', 0))],
            })

            # Scale numeric
            numeric_scaled = pd.DataFrame(
                scaler.transform(numeric_df[['MonthlyCharges_yeojohnson', 'TotalCharges_knn_yeojohnson']]),
                columns=['MonthlyCharges_yeojohnson', 'TotalCharges_knn_yeojohnson']
            )
            numeric_df[['MonthlyCharges_yeojohnson', 'TotalCharges_knn_yeojohnson']] = numeric_scaled

            # Prepare final dataframe
            final_df = pd.DataFrame(columns=MODEL_COLUMNS)
            final_df.loc[0, 'SeniorCitizen_yeojohnson'] = numeric_df.loc[0, 'SeniorCitizen_yeojohnson']
            final_df.loc[0, 'tenure_yeojohnson'] = numeric_df.loc[0, 'tenure_yeojohnson']
            final_df.loc[0, 'MonthlyCharges_yeojohnson'] = numeric_df.loc[0, 'MonthlyCharges_yeojohnson']
            final_df.loc[0, 'TotalCharges_knn_yeojohnson'] = numeric_df.loc[0, 'TotalCharges_knn_yeojohnson']

            # Map categorical
            sim_input = user_input.get('sim', '0')
            sim_model_value = SIM_FORM_TO_MODEL.get(sim_input, 0)  # numeric for model
            final_df.loc[0, 'sim'] = sim_model_value
            final_df.loc[0, 'gender_Male'] = GENDER_MAP.get(user_input.get('gender', 'Female'), 0)
            final_df.loc[0, 'Partner_Yes'] = YES_NO_MAP.get(user_input.get('Partner', 'No'), 0)
            final_df.loc[0, 'Dependents_Yes'] = YES_NO_MAP.get(user_input.get('Dependents', 'No'), 0)
            final_df.loc[0, 'PhoneService_Yes'] = YES_NO_MAP.get(user_input.get('PhoneService', 'No'), 0)
            final_df.loc[0, 'MultipleLines_No phone service'] = 1 if user_input.get('MultipleLines') == "No phone service" else 0
            final_df.loc[0, 'MultipleLines_Yes'] = 1 if user_input.get('MultipleLines') == "Yes" else 0
            final_df.loc[0, 'InternetService_Fiber optic'] = 1 if user_input.get('InternetService') == "Fiber optic" else 0
            final_df.loc[0, 'InternetService_No'] = 1 if user_input.get('InternetService') == "No" else 0
            final_df.loc[0, 'OnlineSecurity_Yes'] = YES_NO_MAP.get(user_input.get('OnlineSecurity', 'No'), 0)
            final_df.loc[0, 'OnlineBackup_Yes'] = YES_NO_MAP.get(user_input.get('OnlineBackup', 'No'), 0)
            final_df.loc[0, 'DeviceProtection_Yes'] = YES_NO_MAP.get(user_input.get('DeviceProtection', 'No'), 0)
            final_df.loc[0, 'TechSupport_Yes'] = YES_NO_MAP.get(user_input.get('TechSupport', 'No'), 0)
            final_df.loc[0, 'StreamingTV_Yes'] = YES_NO_MAP.get(user_input.get('StreamingTV', 'No'), 0)
            final_df.loc[0, 'StreamingMovies_Yes'] = YES_NO_MAP.get(user_input.get('StreamingMovies', 'No'), 0)
            final_df.loc[0, 'PaperlessBilling_Yes'] = YES_NO_MAP.get(user_input.get('PaperlessBilling', 'No'), 0)
            final_df.loc[0, 'PaymentMethod_Credit card (automatic)'] = 1 if user_input.get('PaymentMethod') == "Credit card (automatic)" else 0
            final_df.loc[0, 'PaymentMethod_Electronic check'] = 1 if user_input.get('PaymentMethod') == "Electronic check" else 0
            final_df.loc[0, 'PaymentMethod_Mailed check'] = 1 if user_input.get('PaymentMethod') == "Mailed check" else 0
            final_df.loc[0, 'gateway_Enabled'] = YES_NO_MAP.get(user_input.get('gateway', 'No'), 0)
            final_df.loc[0, 'Contract_con'] = int(user_input.get('Contract', 0))

            final_df = final_df[MODEL_COLUMNS]

            # Make prediction
            prediction = model.predict(final_df)[0]

            # Reverse scaling for display
            monthly_min, monthly_max = 18.25, 118.75
            total_min, total_max = 18.8, 8684.8
            raw_monthly = numeric_df["MonthlyCharges_yeojohnson"].values[0] * (monthly_max - monthly_min) + monthly_min
            raw_total = numeric_df["TotalCharges_knn_yeojohnson"].values[0] * (total_max - total_min) + total_min

            # Display SIM as text
            sim_name = SIM_DISPLAY_MAP.get(sim_model_value, "Unknown")

            prediction_text = {
                "sim": sim_name,
                "monthly_charges": f"{raw_monthly:.2f}",
                "total_charges": f"{raw_total:.2f}",
                "result": "CHURN" if prediction == 1 else "STAY"
            }

            return render_template("result.html", prediction_text=prediction_text)

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return render_template("result.html", prediction_text={"sim": "Error", "monthly_charges": "0", "total_charges": "0", "result": "Error during prediction!"})

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
