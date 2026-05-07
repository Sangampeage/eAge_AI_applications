import pandas as pd
import json
import numpy as np
from inference_pipeline import *

df1 = pd.read_csv('solar_data/bengaluru_complete_dataset.csv')
with open('ml_outputs/features_XGBoost_v3_rolling.json') as f:
    features = json.load(f)

print("--- TRAINING DATA ---")
for col in ['ghi_openmeteo', 'clear_sky_ghi', 'hour_cos', 'cos_zenith', 'kt', 'aerosol_optical_depth']:
    print(f"{col}: mean={df1[col].mean():.3f}, max={df1[col].max():.3f}")

print("\n--- INFERENCE DATA ---")
raw = fetch_weather_and_aq()
pv = compute_pvlib(raw)
basic = basic_engineer(pv)
full = advanced_engineer(basic)

target_df = full[full['timestamp'] >= '2026-03-01'].copy()
for col in ['ghi_openmeteo', 'clear_sky_ghi', 'hour_cos', 'cos_zenith', 'kt', 'aerosol_optical_depth']:
    print(f"{col}: mean={target_df[col].mean():.3f}, max={target_df[col].max():.3f}")
