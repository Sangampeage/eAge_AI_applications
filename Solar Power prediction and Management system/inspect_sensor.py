import sqlite3
import pandas as pd
from datetime import datetime

conn = sqlite3.connect('solar_predictions.db')
print("Current Hour:", datetime.now().strftime("%H"))
print("\n--- sensor_data (Latest 5) ---")
df = pd.read_sql('SELECT * FROM sensor_data ORDER BY hour_timestamp DESC LIMIT 5', conn)
print(df)

print("\n--- sensor_raw_readings (Today, Latest 5) ---")
df_raw = pd.read_sql("SELECT * FROM sensor_raw_readings WHERE timestamp LIKE ? ORDER BY timestamp DESC LIMIT 5", conn, params=(datetime.now().strftime("%Y-%m-%d") + "%",))
print(df_raw)

conn.close()
