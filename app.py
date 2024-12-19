import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import xgboost as xgb
from html import escape

app = Flask(__name__)

# Load the XGBoost model from JSON
model = xgb.Booster()

# Use a relative path to load the model file
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
model_path = os.path.join(current_dir, "model.json")      # Construct the full path
model.load_model(model_path)                              # Load the model


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input values from the form
        input_features = [float(x) for x in request.form.values()]
        features_value = np.array(input_features).reshape(1, -1)  # Reshape for a single prediction

        # Define feature names
        features_name = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]

        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(features_value, feature_names=features_name)

        # Make prediction
        output = model.predict(dmatrix)[0]  # Get the first prediction value

        # Interpret the result
        if output < 0.5:  # Assuming binary classification with threshold 0.5
            res_val = "* breast cancer *"
        else:
            res_val = "no breast cancer"

        return render_template('index.html', prediction_text=f'Patient has {escape(res_val)}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {escape(str(e))}")

if __name__ == "__main__":
    app.run(debug=True)
