"""import os
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

# Model expected columns (after all transformations)
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
SIM_MAP = {"1": 0, "2": 1, "3": 2, "4": 3}  # example mapping, adjust if different

# Raw column names for frontend display
RAW_COLUMNS = [
    'gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService',
    'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
    'PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','sim','gateway','tax','Churn'
]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Collect input from form
            user_input = request.form.to_dict()
            logger.info(f"Raw user input: {user_input}")

            # Convert numeric columns
            numeric_df = pd.DataFrame({
                'SeniorCitizen_yeojohnson': [float(user_input.get('SeniorCitizen', 0))],
                'tenure_yeojohnson': [float(user_input.get('tenure', 0))],
                'MonthlyCharges_yeojohnson': [float(user_input.get('MonthlyCharges', 0))],
                'TotalCharges_knn_yeojohnson': [float(user_input.get('TotalCharges', 0))],
            })

            # Apply scaling
            numeric_scaled = pd.DataFrame(
                scaler.transform(numeric_df[['MonthlyCharges_yeojohnson', 'TotalCharges_knn_yeojohnson']]),
                columns=['MonthlyCharges_yeojohnson', 'TotalCharges_knn_yeojohnson']
            )
            numeric_df[['MonthlyCharges_yeojohnson', 'TotalCharges_knn_yeojohnson']] = numeric_scaled

            # Create dataframe for all model columns
            final_df = pd.DataFrame(columns=MODEL_COLUMNS)

            # Populate numerical columns
            final_df.loc[0, 'SeniorCitizen_yeojohnson'] = numeric_df.loc[0, 'SeniorCitizen_yeojohnson']
            final_df.loc[0, 'tenure_yeojohnson'] = numeric_df.loc[0, 'tenure_yeojohnson']
            final_df.loc[0, 'MonthlyCharges_yeojohnson'] = numeric_df.loc[0, 'MonthlyCharges_yeojohnson']
            final_df.loc[0, 'TotalCharges_knn_yeojohnson'] = numeric_df.loc[0, 'TotalCharges_knn_yeojohnson']

            # Map categorical inputs
            final_df.loc[0, 'sim'] = SIM_MAP.get(user_input.get('sim', '0'), 0)
            final_df.loc[0, 'gender_Male'] = GENDER_MAP.get(user_input.get('gender', 'Female'), 0)
            final_df.loc[0, 'Partner_Yes'] = YES_NO_MAP.get(user_input.get('Partner', 'No'), 0)
            final_df.loc[0, 'Dependents_Yes'] = YES_NO_MAP.get(user_input.get('Dependents', 'No'), 0)
            final_df.loc[0, 'PhoneService_Yes'] = YES_NO_MAP.get(user_input.get('PhoneService', 'No'), 0)
            final_df.loc[0, 'MultipleLines_No phone service'] = YES_NO_MAP.get(user_input.get('MultipleLines', 'No phone service'), 0)
            final_df.loc[0, 'MultipleLines_Yes'] = YES_NO_MAP.get(user_input.get('MultipleLines', 'No'), 0)
            final_df.loc[0, 'InternetService_Fiber optic'] = YES_NO_MAP.get(user_input.get('InternetService', 'No'), 0)
            final_df.loc[0, 'InternetService_No'] = YES_NO_MAP.get(user_input.get('InternetService', 'No'), 0)
            final_df.loc[0, 'OnlineSecurity_Yes'] = YES_NO_MAP.get(user_input.get('OnlineSecurity', 'No'), 0)
            final_df.loc[0, 'OnlineBackup_Yes'] = YES_NO_MAP.get(user_input.get('OnlineBackup', 'No'), 0)
            final_df.loc[0, 'DeviceProtection_Yes'] = YES_NO_MAP.get(user_input.get('DeviceProtection', 'No'), 0)
            final_df.loc[0, 'TechSupport_Yes'] = YES_NO_MAP.get(user_input.get('TechSupport', 'No'), 0)
            final_df.loc[0, 'StreamingTV_Yes'] = YES_NO_MAP.get(user_input.get('StreamingTV', 'No'), 0)
            final_df.loc[0, 'StreamingMovies_Yes'] = YES_NO_MAP.get(user_input.get('StreamingMovies', 'No'), 0)
            final_df.loc[0, 'PaperlessBilling_Yes'] = YES_NO_MAP.get(user_input.get('PaperlessBilling', 'No'), 0)
            final_df.loc[0, 'PaymentMethod_Credit card (automatic)'] = YES_NO_MAP.get(user_input.get('PaymentMethod', 'No'), 0)
            final_df.loc[0, 'PaymentMethod_Electronic check'] = YES_NO_MAP.get(user_input.get('PaymentMethod', 'No'), 0)
            final_df.loc[0, 'PaymentMethod_Mailed check'] = YES_NO_MAP.get(user_input.get('PaymentMethod', 'No'), 0)
            final_df.loc[0, 'gateway_Enabled'] = YES_NO_MAP.get(user_input.get('gateway', 'No'), 0)
            final_df.loc[0, 'Contract_con'] = int(user_input.get('Contract', 0))

            # Ensure only model columns are sent
            final_df = final_df[MODEL_COLUMNS]

            # Log for debugging
            logger.info(f"Final dataframe to send to model:\n{final_df.head()}")

            # Make prediction
            prediction = model.predict(final_df)[0]

            # Extract scaled numeric values
            scaled_monthly = final_df["MonthlyCharges_yeojohnson"].values[0]
            scaled_total = final_df["TotalCharges_knn_yeojohnson"].values[0]
            scaled_sim = final_df["sim"].values[0]

            # Reverse scaling
            monthly_min, monthly_max = 18.25, 118.75
            total_min, total_max = 18.8, 8684.8

            raw_monthly = scaled_monthly * (monthly_max - monthly_min) + monthly_min
            raw_total = scaled_total * (total_max - total_min) + total_min

            # Convert SIM number to name
            sim_mapping = {0: "Airtel", 1: "Jio", 2: "BSNL", 3: "VI"}
            sim_name = sim_mapping.get(int(scaled_sim), "Unknown")

            # Build result text
            result_text = "CHURN" if prediction == 1 else "STAY"
            prediction_text = f"""
                #SIM: {sim_name} <br>
                #Monthly Charges: {raw_monthly:.2f} <br>
                #Total Charges: {raw_total:.2f} <br>
                #Prediction: {result_text}


            # Render the result page
            #return render_template("result.html", prediction_text=prediction_text)


        #except Exception as e:
           # logger.error(f"Error during prediction: {e}")
           # return render_template("result.html", prediction_text="Error during prediction!")
    #return render_template("home.html")

#if __name__ == "__main__":
    #app.run(debug=True) """"



"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Churn Prediction</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
</head>
<body>
<div class="container mt-5">
    <h2 class="mb-4">Customer Churn Prediction</h2>
    <form method="POST" action="/">
        <!-- Gender -->
        <div class="mb-3">
            <label for="gender" class="form-label">Gender</label>
            <select class="form-select" name="gender" id="gender">
                <option value="Male">Male</option>
                <option value="Female">Female</option>
            </select>
        </div>

        <!-- SeniorCitizen -->
        <div class="mb-3">
            <label for="SeniorCitizen" class="form-label">Senior Citizen (1 = Yes, 0 = No)</label>
            <input type="number" class="form-control" id="SeniorCitizen" name="SeniorCitizen" min="0" max="1" value="0">
        </div>

        <!-- Partner -->
        <div class="mb-3">
            <label for="Partner" class="form-label">Partner</label>
            <select class="form-select" name="Partner" id="Partner">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- Dependents -->
        <div class="mb-3">
            <label for="Dependents" class="form-label">Dependents</label>
            <select class="form-select" name="Dependents" id="Dependents">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- Tenure -->
        <div class="mb-3">
            <label for="tenure" class="form-label">Tenure (Months)</label>
            <input type="number" class="form-control" id="tenure" name="tenure" min="0" value="0">
        </div>

        <!-- PhoneService -->
        <div class="mb-3">
            <label for="PhoneService" class="form-label">Phone Service</label>
            <select class="form-select" name="PhoneService" id="PhoneService">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- MultipleLines -->
        <div class="mb-3">
            <label for="MultipleLines" class="form-label">Multiple Lines</label>
            <select class="form-select" name="MultipleLines" id="MultipleLines">
                <option value="No">No</option>
                <option value="Yes">Yes</option>
                <option value="No phone service">No phone service</option>
            </select>
        </div>

        <!-- InternetService -->
        <div class="mb-3">
            <label for="InternetService" class="form-label">Internet Service</label>
            <select class="form-select" name="InternetService" id="InternetService">
                <option value="DSL">DSL</option>
                <option value="Fiber optic">Fiber optic</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- OnlineSecurity -->
        <div class="mb-3">
            <label for="OnlineSecurity" class="form-label">Online Security</label>
            <select class="form-select" name="OnlineSecurity" id="OnlineSecurity">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- OnlineBackup -->
        <div class="mb-3">
            <label for="OnlineBackup" class="form-label">Online Backup</label>
            <select class="form-select" name="OnlineBackup" id="OnlineBackup">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- DeviceProtection -->
        <div class="mb-3">
            <label for="DeviceProtection" class="form-label">Device Protection</label>
            <select class="form-select" name="DeviceProtection" id="DeviceProtection">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- TechSupport -->
        <div class="mb-3">
            <label for="TechSupport" class="form-label">Tech Support</label>
            <select class="form-select" name="TechSupport" id="TechSupport">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- StreamingTV -->
        <div class="mb-3">
            <label for="StreamingTV" class="form-label">Streaming TV</label>
            <select class="form-select" name="StreamingTV" id="StreamingTV">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- StreamingMovies -->
        <div class="mb-3">
            <label for="StreamingMovies" class="form-label">Streaming Movies</label>
            <select class="form-select" name="StreamingMovies" id="StreamingMovies">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- Contract -->
        <div class="mb-3">
            <label for="Contract" class="form-label">Contract (0=Month-to-month, 1=One year, 2=Two year)</label>
            <input type="number" class="form-control" name="Contract" id="Contract" min="0" max="2" value="0">
        </div>

        <!-- PaperlessBilling -->
        <div class="mb-3">
            <label for="PaperlessBilling" class="form-label">Paperless Billing</label>
            <select class="form-select" name="PaperlessBilling" id="PaperlessBilling">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <!-- PaymentMethod -->
        <div class="mb-3">
            <label for="PaymentMethod" class="form-label">Payment Method</label>
            <select class="form-select" name="PaymentMethod" id="PaymentMethod">
                <option value="Electronic check">Electronic check</option>
                <option value="Mailed check">Mailed check</option>
                <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                <option value="Credit card (automatic)">Credit card (automatic)</option>
            </select>
        </div>

        <!-- MonthlyCharges -->
        <div class="mb-3">
            <label for="MonthlyCharges" class="form-label">Monthly Charges</label>
            <input type="number" class="form-control" id="MonthlyCharges" name="MonthlyCharges" step="0.01" value="0">
        </div>

        <!-- TotalCharges -->
        <div class="mb-3">
            <label for="TotalCharges" class="form-label">Total Charges</label>
            <input type="number" class="form-control" id="TotalCharges" name="TotalCharges" step="0.01" value="0">
        </div>

        <!-- SIM -->
        <div class="mb-3">
            <label for="sim" class="form-label">SIM</label>
            <select class="form-select" name="sim" id="sim">
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
            </select>
        </div>

        <!-- Gateway -->
        <div class="mb-3">
            <label for="gateway" class="form-label">Gateway Enabled</label>
            <select class="form-select" name="gateway" id="gateway">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>
        </div>

        <button type="submit" class="btn btn-primary">Predict Churn</button>
    </form>
</div>
</body>
</html>
"""


"""<!-- templates/result.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background-color: #fff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            text-align: center;
            width: 400px;
        }
        h1 {
            color: #333;
        }
        p {
            font-size: 18px;
            margin-top: 20px;
        }
        a {
            display: inline-block;
            margin-top: 25px;
            text-decoration: none;
            color: #fff;
            background-color: #007BFF;
            padding: 10px 20px;
            border-radius: 8px;
        }
        a:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Prediction Result</h1>
        <p><p>{{ prediction_text|safe }}</p></p>
        <a href="{{ url_for('home') }}">Go Back</a>
    </div>
</body>
</html>
"""