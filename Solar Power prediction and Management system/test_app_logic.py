import pandas as pd
import sqlite3
from datetime import datetime, timedelta

DB_PATH = 'solar_predictions.db'

def test_loading():
    conn = sqlite3.connect(DB_PATH)
    
    # Load sensors
    print("--- Loading Sensors ---")
    df_sensor = pd.read_sql('SELECT * FROM sensor_data', conn)
    if not df_sensor.empty:
        print("Raw hour_timestamp head:", df_sensor['hour_timestamp'].head())
        df_sensor['hour_timestamp_pd'] = pd.to_datetime(df_sensor['hour_timestamp'])
        print("After pd.to_datetime:", df_sensor['hour_timestamp_pd'].dtype)
        
        # Shift back
        df_sensor['shifted'] = df_sensor['hour_timestamp_pd'] - pd.Timedelta(hours=1)
        
        # Localize
        if getattr(df_sensor['shifted'].dtype, 'tz', None) is not None:
            df_sensor['final'] = df_sensor['shifted'].dt.tz_localize(None)
        else:
            df_sensor['final'] = df_sensor['shifted']
        
        print("Final type:", df_sensor['final'].dtype)
        print("Final head:\n", df_sensor[['hour_timestamp', 'final']].head())
    else:
        print("sensor_data is empty!")

    # Load predictions (mocking one table)
    print("\n--- Loading Predictions ---")
    try:
        df_preds = pd.read_sql('SELECT timestamp FROM predictions_xgboost LIMIT 5', conn)
        df_preds['timestamp_pd'] = pd.to_datetime(df_preds['timestamp'])
        print("Preds timestamp type:", df_preds['timestamp_pd'].dtype)
        if getattr(df_preds['timestamp_pd'].dtype, 'tz', None) is not None:
            df_preds['final'] = df_preds['timestamp_pd'].dt.tz_localize(None)
        else:
            df_preds['final'] = df_preds['timestamp_pd']
        print("Final preds type:", df_preds['final'].dtype)
    except Exception as e:
        print("Preds error:", e)

    conn.close()

test_loading()
