"""
Smart Solar Grid — Complete Data Collection Pipeline
Sources: CAMS + OpenMeteo Weather + OpenMeteo AQ + pvlib
All timestamps in True Solar Time (solar noon = 12:00 always)
"""

import cdsapi
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import pvlib
import os
import time
from io import StringIO
from retry_requests import retry
from pvlib.location import Location


# ─────────────────────────────────────────────────────
# CONFIGURATION — change only this block per location
# ─────────────────────────────────────────────────────
LAT           = 12.9784
LON           = 77.6358
ALTITUDE      = 920          # metres
LOCATION_NAME = 'bengaluru'
START_YEAR    = 2016         # CAMS data for India starts ~2016
END_YEAR      = 2024
OUTPUT_DIR    = 'solar_data'
# ─────────────────────────────────────────────────────


# ═══════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════

def true_solar_time_offset(lon):
    """
    Offset in hours from UTC to True Solar Time.
    True Solar Time = UTC + (longitude / 15)
    Solar noon is always 12:00 in TST regardless of timezone.
    """
    return lon / 15.0


def utc_to_true_solar_time(df, lon, col='timestamp'):
    """Convert UTC timestamps to True Solar Time."""
    offset  = pd.Timedelta(hours=true_solar_time_offset(lon))
    df      = df.copy()
    df[col] = df[col] + offset
    df[col] = df[col].dt.tz_localize(None)
    return df


def get_openmeteo_client():
    """Create a cached + retry OpenMeteo client."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


# ═══════════════════════════════════════════
# SOURCE 3: CAMS RADIATION
# ═══════════════════════════════════════════

def parse_cams_csv(filepath):
    """
    Parse CAMS solar radiation CSV.
    The actual header line is prefixed with '# Observation period'
    so pandas comment='#' would skip it.
    We manually find it and use it as column names.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Find header line — starts with '# Observation period'
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('# Observation period'):
                header_idx = i
                break

        if header_idx is None:
            print(f'    No header found in {filepath}')
            return None

        # Strip '# ' prefix to get clean column names
        header_line = lines[header_idx].strip().lstrip('# ')
        columns     = [col.strip() for col in header_line.split(';')]

        # Parse data lines below header
        data_str = ''.join(lines[header_idx + 1:])
        df = pd.read_csv(
            StringIO(data_str),
            sep=';',
            header=None,
            names=columns
        )

        # Timestamp = END of interval
        # Raw: 2021-01-01T00:00:00.0/2021-01-01T01:00:00.0
        df['timestamp'] = pd.to_datetime(
            df['Observation period']
              .str.split('/')
              .str[1]
              .str.replace('.0', '', regex=False),
            format='%Y-%m-%dT%H:%M:%S'
        )
        # Already in True Solar Time from CAMS — no conversion needed

        # Rename to clean names
        df = df.rename(columns={
            'GHI':           'ghi_cams',
            'DHI':           'dhi_cams',
            'BNI':           'dni_cams',
            'BHI':           'bhi_cams',
            'Clear sky GHI': 'clear_sky_ghi_cams',
            'Clear sky BNI': 'clear_sky_dni_cams',
            'Clear sky DHI': 'clear_sky_dhi_cams',
            'TOA':           'toa_irradiance',
            'Reliability':   'cams_reliability'
        })

        # Keep only acceptable quality rows
        df = df[df['cams_reliability'] >= 0.8].copy()

        # Keep required columns only
        keep = [
            'timestamp',
            'ghi_cams',
            'dhi_cams',
            'dni_cams',
            'bhi_cams',
            'clear_sky_ghi_cams',
            'clear_sky_dni_cams',
            'clear_sky_dhi_cams',
            'toa_irradiance'
        ]
        return df[keep].reset_index(drop=True)

    except Exception as e:
        print(f'    Parse error in {filepath}: {e}')
        return None


def fetch_cams(lat, lon, location_name, start_year, end_year, output_dir):
    """
    Download CAMS solar radiation year by year.
    Skips years already downloaded.
    Handles no-data errors gracefully.
    """
    os.makedirs(f'{output_dir}/cams_raw', exist_ok=True)
    client  = cdsapi.Client()
    dataset = "cams-solar-radiation-timeseries"
    all_dfs = []

    for year in range(start_year, end_year + 1):
        raw_file = f'{output_dir}/cams_raw/cams_{location_name}_{year}.csv'

        if not os.path.exists(raw_file):
            print(f'  Downloading {year}...')
            request = {
                "sky_type":       "observed_cloud",
                "location":       {"longitude": lon, "latitude": lat},
                "altitude":       ["-999"],
                "date":           [f"{year}-01-01/{year}-12-31"],
                "time_step":      "1hour",
                "time_reference": "true_solar_time",
                "data_format":    "csv"
            }
            try:
                client.retrieve(dataset, request).download(raw_file)
                print(f'    Saved: {raw_file}')
            except Exception as e:
                if "no data available" in str(e).lower():
                    print(f'    {year}: No CAMS data at this location — skipping')
                else:
                    print(f'    {year}: Download error → {e}')
                continue
            time.sleep(2)
        else:
            print(f'  {year}: Already downloaded')

        df = parse_cams_csv(raw_file)
        if df is not None:
            all_dfs.append(df)
            print(f'    {year}: {len(df):,} rows parsed')
        else:
            print(f'    {year}: Parse failed — check file manually')

    if not all_dfs:
        print('[CAMS] No data collected — returning None')
        return None

    cams_df = pd.concat(all_dfs, ignore_index=True)
    cams_df = cams_df.sort_values('timestamp').reset_index(drop=True)
    print(f'[CAMS] Total: {len(cams_df):,} rows | '
          f'{cams_df["timestamp"].min()} to {cams_df["timestamp"].max()}')
    return cams_df


# ═══════════════════════════════════════════
# SOURCE 1: OPENMETEO WEATHER
# ═══════════════════════════════════════════

def fetch_openmeteo_weather(lat, lon, location_name,
                             start_year, end_year, output_dir):
    """
    Download OpenMeteo historical weather variables.
    Collected in UTC then converted to True Solar Time.
    """
    os.makedirs(f'{output_dir}/openmeteo_raw', exist_ok=True)
    raw_file = f'{output_dir}/openmeteo_raw/weather_{location_name}.csv'

    if os.path.exists(raw_file):
        print('  Already downloaded — loading from file')
        df = pd.read_csv(raw_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f'  Loaded: {len(df):,} rows')
        return df

    print('  Downloading...')
    openmeteo = get_openmeteo_client()

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": f"{start_year}-01-01",
        "end_date":   f"{end_year}-12-31",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "total_column_integrated_water_vapour",
            "precipitation",
            "shortwave_radiation",
            "direct_normal_irradiance",
            "diffuse_radiation",
            "sunshine_duration",
            "is_day"
        ],
        "wind_speed_unit": "ms",
        "timezone":        "UTC"
    }

    responses = openmeteo.weather_api(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params
    )
    response = responses[0]
    hourly   = response.Hourly()

    timestamps = pd.date_range(
        start     = pd.to_datetime(hourly.Time(),    unit="s", utc=True),
        end       = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=hourly.Interval()),
        inclusive = "left"
    )

    col_names = [
        "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
        "wind_speed_10m", "wind_direction_10m", "surface_pressure",
        "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
        "cloud_cover_high", "total_column_integrated_water_vapour",
        "precipitation", "ghi_openmeteo", "dni_openmeteo",
        "dhi_openmeteo", "sunshine_duration", "is_day"
    ]

    data = {"timestamp": timestamps}
    for i, name in enumerate(col_names):
        data[name] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df = utc_to_true_solar_time(df, lon)
    df.to_csv(raw_file, index=False)
    print(f'  {len(df):,} rows saved (True Solar Time)')
    return df


# ═══════════════════════════════════════════
# SOURCE 2: OPENMETEO AIR QUALITY
# ═══════════════════════════════════════════

def fetch_openmeteo_airquality(lat, lon, location_name,
                                start_year, end_year, output_dir):
    """
    Download OpenMeteo Air Quality — aerosol, dust, PM.
    Global data reliably available from August 2022.
    Earlier rows will be filled in handle_missing_aerosol().
    Collected in UTC then converted to True Solar Time.
    """
    os.makedirs(f'{output_dir}/openmeteo_raw', exist_ok=True)
    raw_file = f'{output_dir}/openmeteo_raw/airquality_{location_name}.csv'

    if os.path.exists(raw_file):
        print('  Already downloaded — loading from file')
        df = pd.read_csv(raw_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f'  Loaded: {len(df):,} rows')
        return df

    print('  Downloading...')
    aq_start  = max(start_year, 2013)
    openmeteo = get_openmeteo_client()

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": f"{aq_start}-01-01",
        "end_date":   f"{end_year}-12-31",
        "hourly": [
            "aerosol_optical_depth",
            "dust",
            "pm10",
            "pm2_5",
            "uv_index"
        ],
        "timezone": "UTC"
    }

    responses = openmeteo.weather_api(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params=params
    )
    response = responses[0]
    hourly   = response.Hourly()

    timestamps = pd.date_range(
        start     = pd.to_datetime(hourly.Time(),    unit="s", utc=True),
        end       = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq      = pd.Timedelta(seconds=hourly.Interval()),
        inclusive = "left"
    )

    col_names = ["aerosol_optical_depth", "dust", "pm10", "pm2_5", "uv_index"]

    data = {"timestamp": timestamps}
    for i, name in enumerate(col_names):
        data[name] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    df = utc_to_true_solar_time(df, lon)
    df.to_csv(raw_file, index=False)
    print(f'  {len(df):,} rows saved (True Solar Time)')
    return df


# ═══════════════════════════════════════════
# AEROSOL MISSING DATA HANDLER
# ═══════════════════════════════════════════

def handle_missing_aerosol(df):
    """
    Handle missing aerosol values (before Aug 2022 globally).
    Step 1: Flag real vs estimated rows
    Step 2: Fill with climatological mean (same month + hour)
    Step 3: Any remaining NaN filled with safe physical defaults
    """
    print('  Handling missing aerosol data...')

    aerosol_cols = [
        'aerosol_optical_depth',
        'dust',
        'pm10',
        'pm2_5',
        'uv_index'
    ]

    df = df.copy()

    # Step 1: Flag before filling
    # 1 = real measurement from API, 0 = will be estimated
    df['aerosol_data_real'] = (
        df['aerosol_optical_depth'].notna()
    ).astype(int)

    real_count = df['aerosol_data_real'].sum()
    est_count  = (df['aerosol_data_real'] == 0).sum()
    print(f'    Real aerosol rows:    {real_count:,}')
    print(f'    Missing aerosol rows: {est_count:,} will be filled')

    if est_count == 0:
        print('    No missing aerosol data — nothing to fill')
        return df

    # Step 2: Climatological fill using month + hour grouping
    df['_month'] = df['timestamp'].dt.month
    df['_hour']  = df['timestamp'].dt.hour

    for col in aerosol_cols:
        if col not in df.columns:
            continue

        missing_mask = df[col].isna()
        if missing_mask.sum() == 0:
            continue

        # Mean per (month, hour) computed from real data only
        clim = (
            df[df[col].notna()]
            .groupby(['_month', '_hour'])[col]
            .mean()
        )

        if clim.empty:
            continue

        # Apply climatology fill
        for idx in df[missing_mask].index:
            key = (df.at[idx, '_month'], df.at[idx, '_hour'])
            if key in clim.index:
                df.at[idx, col] = clim[key]

        newly_filled = missing_mask.sum() - df[col].isna().sum()
        print(f'    {col}: filled {newly_filled:,} with climatology')

    df = df.drop(columns=['_month', '_hour'])

    # Step 3: Safe physical defaults for any still-missing
    defaults = {
        'aerosol_optical_depth': 0.15,
        'dust':                  0.0,
        'pm10':                  20.0,
        'pm2_5':                 10.0,
        'uv_index':              0.0
    }
    for col, default in defaults.items():
        if col in df.columns:
            remaining = df[col].isna().sum()
            if remaining > 0:
                df[col] = df[col].fillna(default)
                print(f'    {col}: filled {remaining:,} remaining with default {default}')

    # Confirm no NaN remains
    total_missing = sum(
        df[col].isna().sum() for col in aerosol_cols if col in df.columns
    )
    if total_missing == 0:
        print('    All aerosol NaN resolved')
    else:
        print(f'    WARNING: {total_missing} NaN still remain after all fill steps')

    return df


# ═══════════════════════════════════════════
# SOURCE 4: pvlib SOLAR GEOMETRY
# ═══════════════════════════════════════════

def compute_pvlib_features(df, lat, lon, altitude):
    """
    Compute solar geometry features using pvlib.
    Input timestamps are in True Solar Time.
    pvlib needs UTC — we convert back, compute, then results
    are pure geometry values not affected by timezone.
    """
    print('  Computing solar geometry...')

    # TST → UTC for pvlib
    offset_hours = true_solar_time_offset(lon)
    times_utc = (
        pd.DatetimeIndex(df['timestamp'])
          .tz_localize('UTC') - pd.Timedelta(hours=offset_hours)
    )

    loc     = Location(lat, lon, altitude=altitude, tz='UTC')
    solpos  = loc.get_solarposition(times_utc)
    clearsk = loc.get_clearsky(times_utc, model='ineichen')

    df = df.copy()
    df['solar_zenith']    = solpos['zenith'].values
    df['solar_azimuth']   = solpos['azimuth'].values
    df['solar_elevation'] = solpos['elevation'].values
    df['cos_zenith']      = np.cos(np.radians(df['solar_zenith'])).clip(lower=0)
    df['clear_sky_ghi']   = clearsk['ghi'].values
    df['clear_sky_dni']   = clearsk['dni'].values
    df['clear_sky_dhi']   = clearsk['dhi'].values
    df['is_day_pvlib']    = (solpos['elevation'].values > 0).astype(int)

    print(f'  Solar geometry done — {len(df):,} rows')
    return df


# ═══════════════════════════════════════════
# ENGINEERED FEATURES
# ═══════════════════════════════════════════

def compute_days_since_rain(precip_series, threshold=5.0):
    """
    Count hours since last rainfall above threshold mm.
    Expressed as days (divide by 24).
    Resets to 0 when rainfall exceeds threshold.
    """
    result = []
    count  = 0
    for p in precip_series:
        if p >= threshold:
            count = 0
        else:
            count += 1
        result.append(count / 24.0)
    return result


def engineer_features(df):
    """
    Compute all engineered features from collected data.
    Timestamps are in True Solar Time — solar noon = 12:00 always.
    """
    print('  Engineering features...')
    df = df.copy()

    # Time features in True Solar Time
    hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    doy  = df['timestamp'].dt.dayofyear

    df['hour_sin']  = np.sin(2 * np.pi * hour / 24)
    df['hour_cos']  = np.cos(2 * np.pi * hour / 24)
    df['day_sin']   = np.sin(2 * np.pi * doy  / 365.25)
    df['day_cos']   = np.cos(2 * np.pi * doy  / 365.25)
    df['hour']      = df['timestamp'].dt.hour
    df['month']     = df['timestamp'].dt.month
    df['year']      = df['timestamp'].dt.year

    # Clearness index — actual vs theoretical clear sky
    df['kt'] = (
        df['ghi_openmeteo'] / df['clear_sky_ghi'].replace(0, np.nan)
    ).clip(0, 1.5).fillna(0)

    # Physics anchor feature
    df['ghi_clear_weighted'] = df['clear_sky_ghi'] * df['cos_zenith']

    # Normalized CAMS target (primary training target)
    df['csi'] = (
        df['ghi_cams'] / df['clear_sky_ghi'].replace(0, np.nan)
    ).clip(0, 1.5).fillna(0)

    # Dust accumulation proxy
    df['days_since_rain_5mm'] = compute_days_since_rain(df['precipitation'])

    print(f'  Feature engineering done — {df.shape[1]} total columns')
    return df


# ═══════════════════════════════════════════
# MERGE ALL SOURCES
# ═══════════════════════════════════════════

def merge_all_sources(cams_df, weather_df, aq_df):
    """
    Merge all sources on True Solar Time timestamp.
    Rounds to nearest hour to handle minor offsets.
    CAMS is the anchor — ground truth training target.
    """
    print('  Merging sources...')

    if cams_df is None:
        print('  ERROR: CAMS data is None — target variable missing')
        return None

    for df in [cams_df, weather_df, aq_df]:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('h')

    # Inner join: only rows where CAMS + weather both present
    merged = cams_df.merge(weather_df, on='timestamp', how='inner')

    # Left join AQ: missing rows handled in handle_missing_aerosol
    merged = merged.merge(aq_df, on='timestamp', how='left')

    print(f'  CAMS rows:       {len(cams_df):,}')
    print(f'  Weather rows:    {len(weather_df):,}')
    print(f'  AQ rows:         {len(aq_df):,}')
    print(f'  After merge:     {len(merged):,}')
    return merged


# ═══════════════════════════════════════════
# QUALITY CHECK
# ═══════════════════════════════════════════

def quality_check(df):
    """Print summary statistics of the final dataset."""
    print('\n[Quality Check]')
    print(f'  Total rows:                {len(df):,}')
    print(f'  Date range:                {df["timestamp"].min()} → {df["timestamp"].max()}')
    print(f'  Total columns:             {df.shape[1]}')
    print(f'  Daytime rows:              {(df["is_day_pvlib"]==1).sum():,}')
    print(f'  Nighttime rows:            {(df["is_day_pvlib"]==0).sum():,}')
    print(f'  Missing ghi_cams:          {df["ghi_cams"].isna().sum()}')
    print(f'  Missing temperature_2m:    {df["temperature_2m"].isna().sum()}')
    print(f'  Missing AOD (after fill):  {df["aerosol_optical_depth"].isna().sum()}')
    print(f'  Real aerosol rows:         {df["aerosol_data_real"].sum():,}')
    print(f'  Estimated aerosol rows:    {(df["aerosol_data_real"]==0).sum():,}')
    print(f'  Max GHI CAMS (Wh/m2):     {df["ghi_cams"].max():.1f}')
    print(f'  Max GHI OpenMeteo:         {df["ghi_openmeteo"].max():.1f}')
    print(f'  Max CSI:                   {df["csi"].max():.3f}')
    return df


# ═══════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_file = f'{OUTPUT_DIR}/{LOCATION_NAME}_complete_dataset.csv'

    # Load and return if already complete
    if os.path.exists(final_file):
        print(f'Final dataset already exists: {final_file}')
        df = pd.read_csv(final_file, parse_dates=['timestamp'])
        print(f'Loaded: {len(df):,} rows, {df.shape[1]} columns')
        return df

    # Step 1: CAMS ground truth GHI
    print('\n=== STEP 1: CAMS Radiation ===')
    cams_df = fetch_cams(LAT, LON, LOCATION_NAME, START_YEAR, END_YEAR, OUTPUT_DIR)

    # Step 2: OpenMeteo weather variables
    print('\n=== STEP 2: OpenMeteo Weather ===')
    weather_df = fetch_openmeteo_weather(LAT, LON, LOCATION_NAME, START_YEAR, END_YEAR, OUTPUT_DIR)

    # Step 3: OpenMeteo air quality / aerosol
    print('\n=== STEP 3: OpenMeteo Air Quality ===')
    aq_df = fetch_openmeteo_airquality(LAT, LON, LOCATION_NAME, START_YEAR, END_YEAR, OUTPUT_DIR)

    # Step 4: Merge all on True Solar Time timestamp
    print('\n=== STEP 4: Merge Sources ===')
    merged = merge_all_sources(cams_df, weather_df, aq_df)
    if merged is None:
        print('Pipeline stopped — CAMS data required')
        return None

    # Step 4B: Fill missing aerosol before Aug 2022
    print('\n=== STEP 4B: Fill Missing Aerosol ===')
    merged = handle_missing_aerosol(merged)

    # Step 5: pvlib solar geometry
    print('\n=== STEP 5: pvlib Solar Geometry ===')
    merged = compute_pvlib_features(merged, LAT, LON, ALTITUDE)

    # Step 6: Engineered features
    print('\n=== STEP 6: Feature Engineering ===')
    merged = engineer_features(merged)

    # Step 7: Quality check
    quality_check(merged)

    # Step 8: Save
    merged.to_csv(final_file, index=False)
    print(f'\nFinal dataset saved: {final_file}')
    print(f'Shape: {merged.shape}')
    print('\nAll columns:')
    for i, col in enumerate(merged.columns, 1):
        print(f'  {i:02d}. {col}')

    return merged


if __name__ == '__main__':
    df = run_pipeline()