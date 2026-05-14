import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os
import json
import logging
from pathlib import Path

from preprocessing import Preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow container to override paths via env vars
_DATA_DIR_ENV = os.environ.get("DATA_DIR")
if _DATA_DIR_ENV:
    DATA_PATH = Path(_DATA_DIR_ENV) / "sensor_Crop_Dataset (1).csv"
else:
    # Fallback to local structure
    DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sensor_Crop_Dataset (1).csv"

_MODEL_DIR_ENV = os.environ.get("MODEL_DIR")
if _MODEL_DIR_ENV:
    _ARTIFACTS_DIR = Path(_MODEL_DIR_ENV)
else:
    _ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

df = pd.read_csv(DATA_PATH)

pre = Preprocessor()
pre.fit(df)

X = pre.transform(df)
y = pre.transform_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- RF ----------------
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(rf, _ARTIFACTS_DIR / "rf_model.pkl")

# ---------------- XGB ----------------
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    objective="multi:softprob",
    eval_metric="mlogloss"
)
xgb.fit(X_train, y_train)

joblib.dump(xgb, _ARTIFACTS_DIR / "xgb_model.pkl")

# ---------------- Evaluation ----------------
def top_k_accuracy(model, X, y, k=3):
    probs = model.predict_proba(X)
    top_k = np.argsort(probs, axis=1)[:, -k:]
    return np.mean([y[i] in top_k[i] for i in range(len(y))])

try:
    results = {
        "status": "success",
        "message": "Model training completed successfully.",
        "data": {
            "RandomForest": {
                "accuracy": accuracy_score(y_test, rf.predict(X_test)),
                "top3_accuracy": top_k_accuracy(rf, X_test, y_test)
            },
            "XGBoost": {
                "accuracy": accuracy_score(y_test, xgb.predict(X_test)),
                "top3_accuracy": top_k_accuracy(xgb, X_test, y_test)
            }
        }
    }
    logger.info("Training Results: %s", json.dumps(results, indent=2))
except Exception as e:
    logger.error("Error during evaluation: %s", str(e))
    logger.info(json.dumps({
        "status": "failed",
        "message": f"Training failed during evaluation: {str(e)}",
        "data": None
    }, indent=2))