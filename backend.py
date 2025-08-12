# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import xgboost as xgb  # <-- Import xgboost
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# --- 1. Load the model and training columns ---
# Correctly load the saved XGBoost model and columns
model = xgb.XGBRegressor()
model.load_model("best_xgb_model.json") # <-- Correct filename and load method
training_columns = joblib.load('training_columns.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # --- 2. Get data from the POST request ---
    data = request.get_json()
    
    # --- 3. Preprocess the input data ---
    input_df = pd.DataFrame([data])
    
    for col in training_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[training_columns]

    # --- 4. Make a prediction ---
    prediction = model.predict(input_df)
    
    # --- 5. Return the result ---
    return jsonify({'prediction': prediction.tolist()[0]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)