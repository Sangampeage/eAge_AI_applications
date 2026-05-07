import sqlite3
conn = sqlite3.connect('solar_predictions.db')
print("Schema:")
print(conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='predictions_xgboost'").fetchone()[0])
print("\nDuplicate Check:")
res = conn.execute("SELECT timestamp, COUNT(*) FROM predictions_xgboost GROUP BY timestamp HAVING COUNT(*) > 1 LIMIT 5").fetchall()
print(res)
conn.close()
