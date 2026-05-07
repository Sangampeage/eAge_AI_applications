import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Preprocessor:
    def __init__(self):
        self.soil_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.label_encoder = LabelEncoder()

    def fit(self, df):
        df["MOISTURE"] = df["RELATIVE_HUMIDITY"]

        self.soil_encoder.fit(df[["SOIL"]])
        self.label_encoder.fit(df["CROPS"])

        os.makedirs("artifacts", exist_ok=True)

        joblib.dump(self.soil_encoder, "artifacts/soil_encoder.pkl")
        joblib.dump(self.label_encoder, "artifacts/label_encoder.pkl")
    def transform(self, df):
        soil_encoded = self.soil_encoder.transform(df[["SOIL"]])

        numeric = df[["N","P","K","SOIL_PH","TEMP","MOISTURE"]].values

        return np.concatenate([soil_encoded, numeric], axis=1)

    def transform_target(self, df):
        return self.label_encoder.transform(df["CROPS"])


def load_encoders():
    soil_encoder = joblib.load("artifacts/soil_encoder.pkl")
    label_encoder = joblib.load("artifacts/label_encoder.pkl")
    return soil_encoder, label_encoder