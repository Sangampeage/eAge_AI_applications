import pandas as pd
import numpy as np
import json
import datetime
import joblib
from collections import deque
import time
import os
import streamlit as st

# =========================
# Predictor Class
# =========================
class RealTimeGlucosePredictor:
    def __init__(self, history_len=12):  # 12 readings = 60 min at 5-min intervals
        self.history_len = history_len
        self.glucose_window = deque([0]*history_len, maxlen=history_len)
        self.last_basal_rate = 1.0  # default basal if unknown
        self.cls_model = joblib.load("xgb_bolus_classifier.pkl")
        self.reg_model = joblib.load("xgb_bolus_regressor.pkl")
        self.feature_cols = joblib.load("bolus_features.pkl")

    def update(self, glucose_value, timestamp=None, basal_rate=None):
        # Update glucose sliding window
        self.glucose_window.appendleft(glucose_value)

        if basal_rate is not None:
            self.last_basal_rate = basal_rate

        # Timestamp
        now = pd.to_datetime(timestamp) if timestamp else datetime.datetime.now()

        # Feature engineering
        features = {}
        for i, val in enumerate(self.glucose_window, start=1):
            features[f"glucose_t-{i*5}min"] = val
        features["basal_rate"] = self.last_basal_rate
        features["glucose_slope"] = (self.glucose_window[0] - self.glucose_window[-1]) / max(1, self.history_len*5)

        features["time_of_day_morning"] = int(5 <= now.hour < 11)
        features["time_of_day_afternoon"] = int(11 <= now.hour < 17)
        features["time_of_day_evening"] = int(17 <= now.hour < 23)
        features["time_of_day_night"] = int(now.hour >= 23 or now.hour < 5)

        for col in self.feature_cols:
            if col not in features:
                features[col] = 0
        X = pd.DataFrame([features])[self.feature_cols]

        # Predictions
        bolus_needed = self.cls_model.predict(X)[0]
        bolus_amount = 0
        if bolus_needed == 1:
            bolus_amount = self.reg_model.predict(X)[0]

        return bolus_needed, bolus_amount, X

# =========================
# Streamlit UI
# =========================
def app():
    st.title("💉 Real-Time Glucose & Insulin Advisor")

    predictor = RealTimeGlucosePredictor(history_len=12)
    filename = "glucose_data.json"

    # placeholders for live updates
    glucose_placeholder = st.empty()
    bolus_placeholder = st.empty()
    basal_placeholder = st.empty()
    features_placeholder = st.empty()

    st.info("Waiting for new glucose readings...")

    while True:
        try:
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                with open(filename, "r") as f:
                    last_line = f.readlines()[-1].strip()
                if not last_line:
                    continue

                data = json.loads(last_line)
                glucose = data.get("Glucose")
                timestamp = data.get("time")
                basal_rate = data.get("basal_rate", None)

                if glucose is not None:
                    bolus_needed, bolus_amount, features_used = predictor.update(glucose, timestamp, basal_rate)

                    # Update UI
                    glucose_placeholder.metric("Current Glucose", f"{glucose} mg/dL", timestamp)
                    bolus_placeholder.metric("Bolus Prediction", 
                        f"{'Yes' if bolus_needed else 'No'}", 
                        f"{round(bolus_amount,2)} units" if bolus_needed else "0 units")
                    basal_placeholder.metric("Basal Rate", f"{predictor.last_basal_rate} U/hr")
                    features_placeholder.json(features_used.to_dict(orient="records")[0])

            time.sleep(5)  # check every 5 sec
        except Exception as e:
            st.error(f"Error reading file: {e}")
            time.sleep(5)

if __name__ == "__main__":
    app()
