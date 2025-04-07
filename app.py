from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import time
import json
import requests
import numpy as np

app = Flask(__name__)

model = None
features = None

MODEL_PATH = os.path.join("model", "sales_model.pkl")
FEATURES_PATH = os.path.join("model", "features.json")

# Optional: download large model if not found
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        model_url = "https://drive.google.com/file/d/1_xw7k3n3EhSGWfIbOs0eOw2PNM8SKRWW/view?usp=sharing" 
        response = requests.get(model_url)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded.")

def load_model_and_features():
    global model, features

    if model is None:
        download_model_if_needed()

        start = time.time()
        model = joblib.load(MODEL_PATH)
        end = time.time()
        print(f"✅ Model loaded in {end - start:.2f} seconds.")

    if features is None:
        with open(FEATURES_PATH) as f:
            features = json.load(f)
        print("✅ Features loaded.")

    return model, features

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    time_taken = None

    if request.method == "POST":
        try:
            start_time = time.time()
            model, features = load_model_and_features()

            # Gather form data
            store = int(request.form["store"])
            year = int(request.form["year"])
            month = int(request.form["month"])
            day = int(request.form["day"])
            promo = int(request.form["promo"])
            school_holiday = int(request.form["school_holiday"])
            competition_distance = float(request.form["competition_distance"])
            store_type = int(request.form["store_type"])
            assortment = int(request.form["assortment"])

            week_of_year = (month - 1) * 4 + (day // 7)
            day_of_week = 3  # Placeholder: replace with actual logic if needed

            input_data = pd.DataFrame([{
                "Store": store,
                "DayOfWeek": day_of_week,
                "Promo": promo,
                "SchoolHoliday": school_holiday,
                "Year": year,
                "Month": month,
                "Day": day,
                "WeekOfYear": week_of_year,
                "StoreType": store_type,
                "Assortment": assortment,
                "CompetitionDistance": competition_distance
            }])

            # Reorder columns to match training
            input_data = input_data[features]

            # Predict sales and get multiple predictions for confidence
            predictions = model.predict(input_data)
            confidence = np.std(predictions)  # Standard deviation as a confidence measure
            prediction = predictions[0]
            time_taken = round(time.time() - start_time, 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"
            confidence = None

    return render_template("index.html", prediction=prediction, confidence=confidence, time_taken=time_taken)

if __name__ == "__main__":
    app.run(debug=True)
