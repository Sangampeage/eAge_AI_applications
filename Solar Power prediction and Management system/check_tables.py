import sqlite3
import pandas as pd

def check_everything():
    conn = sqlite3.connect('solar_predictions.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    
    # Print tables on separate lines
    print("Tables in DB:")
    for t in tables:
        print(" -", t)
    
    print()
    models = ['xgboost', 'lightgbm', 'randomforest', 'extratrees', 'svr']
    for model in models:
        table_name = f'predictions_{model}'
        if table_name in tables:
            count = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            sample = cursor.execute(f"SELECT timestamp FROM {table_name} ORDER BY timestamp DESC LIMIT 1").fetchone()
            print(f"  {table_name}: {count} rows, latest: {sample[0] if sample else 'N/A'}")
        else:
            print(f"  {table_name}: MISSING!")
    
    print()
    if 'sensor_data' in tables:
        count = cursor.execute("SELECT COUNT(*) FROM sensor_data").fetchone()[0]
        latest = cursor.execute("SELECT hour_timestamp FROM sensor_data ORDER BY hour_timestamp DESC LIMIT 1").fetchone()
        print(f"sensor_data: {count} rows, latest: {latest[0] if latest else 'N/A'}")
        print()
        print("All sensor_data rows (hour_timestamp only):")
        for row in cursor.execute("SELECT hour_timestamp, ghi_avg FROM sensor_data ORDER BY hour_timestamp ASC"):
            print(f"  {row[0]}  ghi_avg={row[1]}")
    else:
        print("sensor_data: MISSING!")
    
    # Check what "sensor_hourly_" might be
    for t in tables:
        if 'sensor' in t.lower() and t != 'sensor_data' and t != 'sensor_raw_readings':
            count = cursor.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
            print(f"\nOther sensor table '{t}': {count} rows")
    
    conn.close()

check_everything()
