import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import pvlib
import joblib
from datetime import datetime, timedelta, date
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pvlib.location import Location

# Configuration
LAT = 12.9784
LON = 77.6358
ALTITUDE = 920
OUTPUT_DIR = 'ml_outputs'
DB_PATH = 'solar_predictions.db'

# ── Dynamic date window ────────────────────────────────────────────────────
# Default: fetch last 60 days (for lag context) + 7 days of future forecast
# Override by passing --start YYYY-MM-DD --end YYYY-MM-DD on command line
# for backfill runs (e.g. Jan 1 2025 → Dec 31 2025).
_today = date.today()
DEFAULT_START = (_today - timedelta(days=60)).isoformat()  # 60-day lookback
DEFAULT_END   = (_today + timedelta(days=6)).isoformat()   # 6-day forecast
# ──────────────────────────────────────────────────────────────────────────

def get_openmeteo_client():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)

def true_solar_time_offset(lon):
    return lon / 15.0

def utc_to_true_solar_time(df, lon):
    offset = pd.Timedelta(hours=true_solar_time_offset(lon))
    df = df.copy()
    df['timestamp'] = df['timestamp'] + offset
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df

def fetch_weather_and_aq(start_date: str, end_date: str):
    print(f"Fetching OpenMeteo Data: {start_date} -> {end_date}")
    client = get_openmeteo_client()

    # Choose API based on date
    # Forecast API (v1/forecast) allows current day + future + ~10 days past
    # Archive API (v1/archive) is for historical data
    _is_historical = datetime.strptime(end_date, "%Y-%m-%d").date() < _today - timedelta(days=5)
    
    base_weather = "https://archive-api.open-meteo.com/v1/archive" if _is_historical else "https://api.open-meteo.com/v1/forecast"
    base_aq      = "https://air-quality-api.open-meteo.com/v1/air-quality" # AQ API handles both
    
    print(f"  [API ] Using {'Archive' if _is_historical else 'Forecast'} for weather")
    
    # 1. Weather
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
            "wind_speed_10m", "wind_direction_10m", "surface_pressure",
            "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
            "total_column_integrated_water_vapour", "precipitation",
            "shortwave_radiation", "direct_normal_irradiance",
            "diffuse_radiation", "sunshine_duration"
        ],
        "wind_speed_unit": "ms",
        "timezone": "UTC"
    }
    # Archive API doesn't have "is_day" variable in hourly, we'll use pvlib for that anyway
    if not _is_historical:
        weather_params["hourly"].append("is_day")
        
    weather_resp = client.weather_api(base_weather, params=weather_params)[0]
    wh = weather_resp.Hourly()
    wt = pd.date_range(
        start=pd.to_datetime(wh.Time(), unit="s", utc=True),
        end=pd.to_datetime(wh.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=wh.Interval()), inclusive="left"
    )
    
    weather_cols = ["temperature_2m", "relative_humidity_2m", "dewpoint_2m",
                    "wind_speed_10m", "wind_direction_10m", "surface_pressure",
                    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                    "total_column_integrated_water_vapour", "precipitation",
                    "ghi_openmeteo", "dni_openmeteo", "dhi_openmeteo",
                    "sunshine_duration"]
    if not _is_historical:
        weather_cols.append("is_day")
        
    wd = {"timestamp": wt}
    for i, name in enumerate(weather_cols):
        wd[name] = wh.Variables(i).ValuesAsNumpy()
    wdf = utc_to_true_solar_time(pd.DataFrame(wd), LON)
    
    if _is_historical:
        wdf['is_day'] = 0 # Placeholder, will use pvlib anyway
    
    # 2. Air Quality
    aq_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["aerosol_optical_depth", "dust", "pm10", "pm2_5", "uv_index"],
        "timezone": "UTC"
    }
    aq_resp = client.weather_api(base_aq, params=aq_params)[0]
    ah = aq_resp.Hourly()
    at = pd.date_range(
        start=pd.to_datetime(ah.Time(), unit="s", utc=True),
        end=pd.to_datetime(ah.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=ah.Interval()), inclusive="left"
    )
    aq_cols = ["aerosol_optical_depth", "dust", "pm10", "pm2_5", "uv_index"]
    ad = {"timestamp": at}
    for i, name in enumerate(aq_cols):
        ad[name] = ah.Variables(i).ValuesAsNumpy()
    adf = utc_to_true_solar_time(pd.DataFrame(ad), LON)
    
    wdf['timestamp'] = pd.to_datetime(wdf['timestamp']).dt.round('h')
    adf['timestamp'] = pd.to_datetime(adf['timestamp']).dt.round('h')
    
    # Fill remaining NaNs in AQ with safe historical defaults
    adf['aerosol_optical_depth'] = adf['aerosol_optical_depth'].fillna(0.31).clip(upper=0.4)
    adf['dust'] = adf['dust'].fillna(3.32).clip(upper=5.0)
    adf['pm10'] = adf['pm10'].fillna(20.0).clip(upper=25.0)
    adf['pm2_5'] = adf['pm2_5'].fillna(20.48).clip(upper=25.0)
    adf['uv_index'] = adf['uv_index'].fillna(0.0)
    
    merged = wdf.merge(adf, on='timestamp', how='left')
    return merged

def compute_pvlib(df):
    offset_hours = true_solar_time_offset(LON)
    times_utc = (pd.DatetimeIndex(df['timestamp']).tz_localize('UTC') - pd.Timedelta(hours=offset_hours))
    loc = Location(LAT, LON, altitude=ALTITUDE, tz='UTC')
    solpos = loc.get_solarposition(times_utc)
    clearsk = loc.get_clearsky(times_utc, model='ineichen')
    
    df['solar_zenith'] = solpos['zenith'].values
    df['solar_azimuth'] = solpos['azimuth'].values
    df['solar_elevation'] = solpos['elevation'].values
    df['cos_zenith'] = np.cos(np.radians(df['solar_zenith'])).clip(lower=0)
    df['clear_sky_ghi'] = clearsk['ghi'].values
    df['clear_sky_dni'] = clearsk['dni'].values
    df['clear_sky_dhi'] = clearsk['dhi'].values
    df['is_day_pvlib'] = (solpos['elevation'].values > 0).astype(int)
    return df

def basic_engineer(df):
    hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    doy = df['timestamp'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * doy / 365.25)
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['kt'] = (df['ghi_openmeteo'] / df['clear_sky_ghi'].replace(0, np.nan)).clip(0, 1.5).fillna(0)
    df['ghi_clear_weighted'] = df['clear_sky_ghi'] * df['cos_zenith']
    
    count = 0
    res = []
    for p in df['precipitation']:
        if pd.isna(p): p = 0
        if p >= 5.0: count = 0
        else: count += 1
        res.append(count / 24.0)
    df['days_since_rain_5mm'] = res
    return df

def advanced_engineer(df):
    # Lags
    for lag in [1, 2, 3, 6, 24, 48, 168]: df[f'ghi_lag_{lag}h'] = df['ghi_openmeteo'].shift(lag)
    for lag in [1, 2, 3, 6, 24]: df[f'cloud_lag_{lag}h'] = df['cloud_cover'].shift(lag)
    for lag in [1, 3, 6, 24]: df[f'aod_lag_{lag}h'] = df['aerosol_optical_depth'].shift(lag)
    for lag in [1, 3, 24]:
        df[f'temp_lag_{lag}h'] = df['temperature_2m'].shift(lag)
        df[f'humid_lag_{lag}h'] = df['relative_humidity_2m'].shift(lag)
    for lag in [1, 2, 3, 24]: df[f'kt_lag_{lag}h'] = df['kt'].shift(lag)
    
    # Rolling
    df['ghi_roll_mean_3h'] = df['ghi_openmeteo'].rolling(3, min_periods=1).mean()
    df['ghi_roll_mean_6h'] = df['ghi_openmeteo'].rolling(6, min_periods=1).mean()
    df['ghi_roll_mean_24h'] = df['ghi_openmeteo'].rolling(24, min_periods=1).mean()
    df['ghi_roll_max_24h'] = df['ghi_openmeteo'].rolling(24, min_periods=1).max()
    df['ghi_roll_std_3h'] = df['ghi_openmeteo'].rolling(3, min_periods=2).std().fillna(0)
    df['ghi_roll_std_24h'] = df['ghi_openmeteo'].rolling(24, min_periods=2).std().fillna(0)
    df['cloud_roll_mean_3h'] = df['cloud_cover'].rolling(3, min_periods=1).mean()
    df['cloud_roll_mean_6h'] = df['cloud_cover'].rolling(6, min_periods=1).mean()
    df['cloud_roll_max_24h'] = df['cloud_cover'].rolling(24, min_periods=1).max()
    df['kt_roll_mean_3h'] = df['kt'].rolling(3, min_periods=1).mean()
    df['kt_roll_std_3h'] = df['kt'].rolling(3, min_periods=2).std().fillna(0)
    df['kt_roll_mean_24h'] = df['kt'].rolling(24, min_periods=1).mean()
    df['precip_sum_6h'] = df['precipitation'].rolling(6, min_periods=1).sum()
    df['precip_sum_24h'] = df['precipitation'].rolling(24, min_periods=1).sum()
    df['precip_sum_72h'] = df['precipitation'].rolling(72, min_periods=1).sum()
    df['aod_roll_mean_24h'] = df['aerosol_optical_depth'].rolling(24, min_periods=1).mean()
    df['aod_roll_max_24h'] = df['aerosol_optical_depth'].rolling(24, min_periods=1).max()
    
    # Sunshine
    df['sunshine_fraction'] = df['sunshine_duration'] / 3600.0
    df['sunshine_lag_1h'] = df['sunshine_duration'].shift(1)
    df['sunshine_lag_3h'] = df['sunshine_duration'].shift(3)
    df['sunshine_roll_mean_3h'] = df['sunshine_duration'].rolling(3, min_periods=1).mean()
    df['sunshine_roll_mean_6h'] = df['sunshine_duration'].rolling(6, min_periods=1).mean()
    
    # Other physics features mentioned
    ghi_safe = df['ghi_openmeteo'].clip(lower=1)
    df['beam_fraction'] = (df['dni_openmeteo'] / ghi_safe).clip(0, 2)
    df['diffuse_fraction'] = (df['dhi_openmeteo'] / ghi_safe).clip(0, 2)
    df['poa_proxy'] = df['ghi_openmeteo'] * df['cos_zenith']
    zenith_rad = np.radians(df['solar_zenith'].clip(0, 89.9))
    df['air_mass'] = (1 / (np.cos(zenith_rad) + 0.001)).clip(1, 40)
    df['turbidity_proxy'] = df['aerosol_optical_depth'] * df['air_mass']
    df['kt_delta_1h'] = df['kt'] - df['kt_lag_1h']
    df['cloud_delta_1h'] = df['cloud_cover'] - df['cloud_lag_1h']
    
    df['humid_cloud'] = df['relative_humidity_2m'] * df['cloud_cover'] / 100
    df['dust_loading'] = df['aerosol_optical_depth'] * (1 - df['precip_sum_24h'].clip(0, 50) / 50)
    df['wind_dust'] = df['wind_speed_10m'] * df['aerosol_optical_depth']
    monthly_mean_temp = df.groupby('month')['temperature_2m'].transform('mean')
    df['temp_departure'] = df['temperature_2m'] - monthly_mean_temp
    df['surface_wet'] = df['precip_sum_24h'] / (df['precip_sum_24h'] + 1)
    df['aerosol_cloud'] = df['aerosol_optical_depth'] * df['cloud_cover'] / 100
    df['zenith_kt'] = df['cos_zenith'] * df['kt']
    df['zenith_cloud'] = (df['cos_zenith'] * (1 - df['cloud_cover'] / 100)).clip(0, 1)
    
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
    df['is_ne_monsoon'] = df['month'].isin([10, 11]).astype(int)
    def season(m):
        if m in [12,1,2]: return 0
        if m in [3,4,5]: return 1
        if m in [6,7,8,9]: return 2
        return 3
    df['season'] = df['month'].apply(season)
    df['is_morning'] = (df['hour'] < 12).astype(int)
    df['solar_noon_dist'] = abs(df['hour'] - 12)
    
    return df

def upsert_predictions(conn, table_name: str, df: pd.DataFrame):
    """
    Upsert (INSERT OR REPLACE) rows into a prediction table.
    Creates the table if it does not exist.
    Preserves all historical rows — only overwrites matching timestamps.
    """
    conn.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp TEXT PRIMARY KEY,
            predicted_ghi REAL
        )
    ''')
    conn.executemany(
        f'INSERT OR REPLACE INTO {table_name} (timestamp, predicted_ghi) VALUES (?, ?)',
        list(zip(df['timestamp'].astype(str), df['predicted_ghi']))
    )
    conn.commit()


def run_inference(start_date: str = None, end_date: str = None):
    """
    Run the full inference pipeline.

    Parameters
    ----------
    start_date : str, optional
        ISO date string (YYYY-MM-DD).  Defaults to 60 days before today.
        Pass a far-back date (e.g. '2025-01-01') to run a historical backfill.
    end_date : str, optional
        ISO date string (YYYY-MM-DD).  Defaults to today + 7 days (forecast).
    """
    if start_date is None:
        start_date = DEFAULT_START
    if end_date is None:
        end_date = DEFAULT_END

    print(f"\n{'='*60}")
    print(f"  Solar GHI Inference Pipeline")
    print(f"  Window : {start_date}  ->  {end_date}")
    print(f"  Today  : {_today.isoformat()}  (predictions up to {end_date})")
    print(f"{'='*60}\n")

    # ── 1. Fetch weather + AQ data ────────────────────────────────────────
    raw   = fetch_weather_and_aq(start_date, end_date)
    pv    = compute_pvlib(raw)
    basic = basic_engineer(pv)
    full  = advanced_engineer(basic)

    # ── 2. Build the prediction target window ─────────────────────────────
    # We keep ALL rows from the fetch window so lags resolve correctly,
    # but we only STORE predictions starting from start_date.
    # The full dataframe is required so rolling/lag features are valid.
    target_df = full[full['timestamp'] >= start_date].copy()
    target_df = target_df.reset_index(drop=True)

    print(f"[Inference] Target rows: {len(target_df):,}  "
          f"({target_df['timestamp'].min()} -> {target_df['timestamp'].max()})")

    # ── 3. Run each model ─────────────────────────────────────────────────
    import json
    models_to_run = ['XGBoost', 'LightGBM', 'RandomForest', 'ExtraTrees', 'SVR']
    results = {'timestamp': target_df['timestamp'].values}

    for model_name in models_to_run:
        model_path = os.path.join(OUTPUT_DIR, f'model_{model_name}_v3_rolling.pkl')
        feat_path  = os.path.join(OUTPUT_DIR, f'features_{model_name}_v3_rolling.json')

        if not os.path.exists(model_path):
            print(f"  [SKIP] {model_name}: model file not found -> {model_path}")
            continue
        if not os.path.exists(feat_path):
            print(f"  [SKIP] {model_name}: features file not found -> {feat_path}")
            continue

        print(f"  [RUN ] {model_name}...")
        with open(feat_path, 'r') as f:
            features = json.load(f)

        model_obj = joblib.load(model_path)
        X = target_df[features].fillna(0).values

        # SVR may have been saved with a scaler
        if isinstance(model_obj, dict) and 'scaler' in model_obj:
            X         = model_obj['scaler'].transform(X)
            predictor = model_obj['model']
        else:
            predictor = model_obj

        preds = predictor.predict(X)

        # Zero out night-time predictions (solar elevation ≤ 0°)
        night_mask         = target_df['is_day_pvlib'].values == 0
        preds[night_mask]  = 0.0
        preds              = np.clip(preds, 0, None)  # no negative GHI

        results[model_name] = preds
        print(f"         peak={preds.max():.1f} W/m²  mean_daytime={preds[~night_mask].mean():.1f} W/m²")

    res_df = pd.DataFrame(results)

    # ── 4. Persist to DB (UPSERT — never wipe history) ───────────────────
    conn = sqlite3.connect(DB_PATH)
    for model_name in models_to_run:
        if model_name not in results:
            continue
        df_model = pd.DataFrame({
            'timestamp'     : res_df['timestamp'],
            'predicted_ghi' : res_df[model_name],
        })
        table = f'predictions_{model_name.lower()}'
        upsert_predictions(conn, table, df_model)
        print(f"  [DB  ] Upserted {len(df_model):,} rows -> {table}")

    # Ensure sensor_data table always exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            hour_timestamp TEXT PRIMARY KEY,
            ghi_avg        REAL,
            ghi_min        REAL,
            ghi_max        REAL,
            sample_count   INTEGER,
            completeness   REAL
        )
    ''')
    conn.commit()
    conn.close()

    print(f"\nSUCCESS: Inference complete. DB: {DB_PATH}")
    print(f"  Window stored: {start_date}  ->  {end_date}")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Solar GHI Inference Pipeline'
    )
    parser.add_argument(
        '--start', default=None,
        help='Start date YYYY-MM-DD (default: today - 60 days)'
    )
    parser.add_argument(
        '--end', default=None,
        help='End date YYYY-MM-DD (default: today + 7 days)'
    )
    args = parser.parse_args()

    run_inference(start_date=args.start, end_date=args.end)
