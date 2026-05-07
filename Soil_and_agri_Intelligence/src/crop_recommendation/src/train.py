import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from preprocessing import Preprocessor

df = pd.read_csv(r"C:\Users\sanga\OneDrive\Desktop\eAge_AI_Applications\Soil_and_agri_Intelligence\data\Crop_recommendation_dataset.csv")

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

joblib.dump(rf, "artifacts/rf_model.pkl")

# ---------------- XGB ----------------
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    objective="multi:softprob",
    eval_metric="mlogloss"
)
xgb.fit(X_train, y_train)

joblib.dump(xgb, "artifacts/xgb_model.pkl")

# ---------------- Evaluation ----------------
def top_k_accuracy(model, X, y, k=3):
    probs = model.predict_proba(X)
    top_k = np.argsort(probs, axis=1)[:, -k:]
    return np.mean([y[i] in top_k[i] for i in range(len(y))])

print("RF Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
print("RF Top3:", top_k_accuracy(rf, X_test, y_test))

print("XGB Accuracy:", accuracy_score(y_test, xgb.predict(X_test)))
print("XGB Top3:", top_k_accuracy(xgb, X_test, y_test))