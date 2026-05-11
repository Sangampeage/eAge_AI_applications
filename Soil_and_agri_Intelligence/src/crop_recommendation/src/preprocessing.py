import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

class Preprocessor:
    def __init__(self):
        self.soil_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.label_encoder = LabelEncoder()

    def fit(self, df):
        df["MOISTURE"] = df["RELATIVE_HUMIDITY"]

        self.soil_encoder.fit(df[["SOIL"]])
        self.label_encoder.fit(df["CROPS"])

        os.makedirs(_ARTIFACTS_DIR, exist_ok=True)

        joblib.dump(self.soil_encoder, os.path.join(_ARTIFACTS_DIR, "soil_encoder.pkl"))
        joblib.dump(self.label_encoder, os.path.join(_ARTIFACTS_DIR, "label_encoder.pkl"))
    def transform(self, df):
        soil_encoded = self.soil_encoder.transform(df[["SOIL"]])

        numeric = df[["N","P","K","SOIL_PH","TEMP","MOISTURE"]].values

        return np.concatenate([soil_encoded, numeric], axis=1)

    def transform_target(self, df):
        return self.label_encoder.transform(df["CROPS"])


def load_encoders():
    soil_encoder = joblib.load(os.path.join(_ARTIFACTS_DIR, "soil_encoder.pkl"))
    label_encoder = joblib.load(os.path.join(_ARTIFACTS_DIR, "label_encoder.pkl"))
    return soil_encoder, label_encoder