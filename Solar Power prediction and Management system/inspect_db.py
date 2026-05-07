import sqlite3
import pandas as pd

conn = sqlite3.connect('solar_predictions.db')

print("--- Tables ---")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for t in tables:
    print(t[0])

print("\n--- Sensor Data Sample ---")
try:
    df_sensor = pd.read_sql("SELECT * FROM sensor_data ORDER BY hour_timestamp DESC LIMIT 10", conn)
    print(df_sensor)
except Exception as e:
    print(f"Error reading sensor_data: {e}")

print("\n--- Predictions Sample (XGBoost) ---")
try:
    df_preds = pd.read_sql("SELECT * FROM predictions_xgboost ORDER BY timestamp DESC LIMIT 5", conn)
    print(df_preds)
except Exception as e:
    print(f"Error reading predictions_xgboost: {e}")

conn.close()
