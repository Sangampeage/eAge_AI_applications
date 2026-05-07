"""
Smart Solar Grid — Multi-Location Data Collection
Collects data for all 35 locations (20 India + 15 Global)
Run this file to collect data for all locations automatically.
Each location saved as: solar_data/{location_name}_complete_dataset.csv
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


# ═══════════════════════════════════════════
# 35 LOCATIONS — 20 INDIA + 15 GLOBAL
# ═══════════════════════════════════════════

LOCATIONS = [

    # ── INDIA: HOT DESERT (Rajasthan/Gujarat) ──────────────────────────
    {
        "name":        "jaisalmer",
        "display":     "Jaisalmer, Rajasthan",
        "lat":         26.9157,
        "lon":         70.9083,
        "altitude":    225,
        "climate":     "Hot Desert",
        "region":      "India"
    },
    {
        "name":        "jodhpur",
        "display":     "Jodhpur, Rajasthan",
        "lat":         26.2389,
        "lon":         73.0243,
        "altitude":    231,
        "climate":     "Hot Desert",
        "region":      "India"
    },
    {
        "name":        "bikaner",
        "display":     "Bikaner, Rajasthan",
        "lat":         28.0229,
        "lon":         73.3119,
        "altitude":    234,
        "climate":     "Hot Desert",
        "region":      "India"
    },
    {
        "name":        "ahmedabad",
        "display":     "Ahmedabad, Gujarat",
        "lat":         23.0225,
        "lon":         72.5714,
        "altitude":    53,
        "climate":     "Semi-Arid",
        "region":      "India"
    },

    # ── INDIA: SEMI-ARID PLATEAU ───────────────────────────────────────
    {
        "name":        "nagpur",
        "display":     "Nagpur, Maharashtra",
        "lat":         21.1458,
        "lon":         79.0882,
        "altitude":    310,
        "climate":     "Semi-Arid Plateau",
        "region":      "India"
    },
    {
        "name":        "hyderabad",
        "display":     "Hyderabad, Telangana",
        "lat":         17.3850,
        "lon":         78.4867,
        "altitude":    536,
        "climate":     "Semi-Arid",
        "region":      "India"
    },
    {
        "name":        "bhopal",
        "display":     "Bhopal, Madhya Pradesh",
        "lat":         23.2599,
        "lon":         77.4126,
        "altitude":    527,
        "climate":     "Tropical Savanna",
        "region":      "India"
    },

    # ── INDIA: COASTAL WEST ────────────────────────────────────────────
    {
        "name":        "mumbai",
        "display":     "Mumbai, Maharashtra",
        "lat":         19.0760,
        "lon":         72.8777,
        "altitude":    11,
        "climate":     "Tropical Wet",
        "region":      "India"
    },
    {
        "name":        "kochi",
        "display":     "Kochi, Kerala",
        "lat":         9.9312,
        "lon":         76.2673,
        "altitude":    3,
        "climate":     "Tropical Rainforest",
        "region":      "India"
    },
    {
        "name":        "mangalore",
        "display":     "Mangalore, Karnataka",
        "lat":         12.9141,
        "lon":         74.8560,
        "altitude":    22,
        "climate":     "Tropical Monsoon",
        "region":      "India"
    },

    # ── INDIA: COASTAL EAST ────────────────────────────────────────────
    {
        "name":        "chennai",
        "display":     "Chennai, Tamil Nadu",
        "lat":         13.0827,
        "lon":         80.2707,
        "altitude":    6,
        "climate":     "Tropical Wet/Dry",
        "region":      "India"
    },
    {
        "name":        "visakhapatnam",
        "display":     "Visakhapatnam, Andhra Pradesh",
        "lat":         17.6868,
        "lon":         83.2185,
        "altitude":    45,
        "climate":     "Tropical Savanna",
        "region":      "India"
    },
    {
        "name":        "bhubaneswar",
        "display":     "Bhubaneswar, Odisha",
        "lat":         20.2961,
        "lon":         85.8245,
        "altitude":    45,
        "climate":     "Tropical Savanna",
        "region":      "India"
    },

    # ── INDIA: GANGETIC PLAIN (High Aerosol) ──────────────────────────
    {
        "name":        "delhi",
        "display":     "New Delhi",
        "lat":         28.6139,
        "lon":         77.2090,
        "altitude":    216,
        "climate":     "Semi-Arid",
        "region":      "India"
    },
    {
        "name":        "lucknow",
        "display":     "Lucknow, Uttar Pradesh",
        "lat":         26.8467,
        "lon":         80.9462,
        "altitude":    123,
        "climate":     "Humid Subtropical",
        "region":      "India"
    },
    {
        "name":        "patna",
        "display":     "Patna, Bihar",
        "lat":         25.5941,
        "lon":         85.1376,
        "altitude":    53,
        "climate":     "Humid Subtropical",
        "region":      "India"
    },

    # ── INDIA: NORTH-EAST ─────────────────────────────────────────────
    {
        "name":        "guwahati",
        "display":     "Guwahati, Assam",
        "lat":         26.1445,
        "lon":         91.7362,
        "altitude":    55,
        "climate":     "Humid Subtropical",
        "region":      "India"
    },

    # ── INDIA: SOUTH PLATEAU ──────────────────────────────────────────
    {
        "name":        "bengaluru",
        "display":     "Bengaluru, Karnataka",
        "lat":         12.9784,
        "lon":         77.6358,
        "altitude":    920,
        "climate":     "Tropical Savanna",
        "region":      "India"
    },
    {
        "name":        "coimbatore",
        "display":     "Coimbatore, Tamil Nadu",
        "lat":         11.0168,
        "lon":         76.9558,
        "altitude":    411,
        "climate":     "Tropical Savanna",
        "region":      "India"
    },

    # ── INDIA: HIGH ALTITUDE ──────────────────────────────────────────
    {
        "name":        "leh",
        "display":     "Leh, Ladakh",
        "lat":         34.1526,
        "lon":         77.5771,
        "altitude":    3524,
        "climate":     "Cold Desert",
        "region":      "India"
    },

    # ── GLOBAL: SIMILAR TO INDIA CLIMATE ──────────────────────────────
    {
        "name":        "riyadh",
        "display":     "Riyadh, Saudi Arabia",
        "lat":         24.7136,
        "lon":         46.6753,
        "altitude":    612,
        "climate":     "Extreme Hot Desert",
        "region":      "Global"
    },
    {
        "name":        "cairo",
        "display":     "Cairo, Egypt",
        "lat":         30.0444,
        "lon":         31.2357,
        "altitude":    23,
        "climate":     "Hot Desert + High AOD",
        "region":      "Global"
    },
    {
        "name":        "karachi",
        "display":     "Karachi, Pakistan",
        "lat":         24.8607,
        "lon":         67.0011,
        "altitude":    13,
        "climate":     "Hot Arid",
        "region":      "Global"
    },
    {
        "name":        "colombo",
        "display":     "Colombo, Sri Lanka",
        "lat":         6.9271,
        "lon":         79.8612,
        "altitude":    7,
        "climate":     "Tropical Rainforest",
        "region":      "Global"
    },

    # ── GLOBAL: TEMPERATE ─────────────────────────────────────────────
    {
        "name":        "madrid",
        "display":     "Madrid, Spain",
        "lat":         40.4168,
        "lon":         -3.7038,
        "altitude":    667,
        "climate":     "Mediterranean",
        "region":      "Global"
    },
    {
        "name":        "rome",
        "display":     "Rome, Italy",
        "lat":         41.9028,
        "lon":         12.4964,
        "altitude":    37,
        "climate":     "Mediterranean",
        "region":      "Global"
    },
    {
        "name":        "sydney",
        "display":     "Sydney, Australia",
        "lat":         -33.8688,
        "lon":         151.2093,
        "altitude":    39,
        "climate":     "Temperate Oceanic",
        "region":      "Global"
    },

    # ── GLOBAL: TROPICAL ──────────────────────────────────────────────
    {
        "name":        "bangkok",
        "display":     "Bangkok, Thailand",
        "lat":         13.7563,
        "lon":         100.5018,
        "altitude":    2,
        "climate":     "Tropical Monsoon",
        "region":      "Global"
    },
    {
        "name":        "kuala_lumpur",
        "display":     "Kuala Lumpur, Malaysia",
        "lat":         3.1390,
        "lon":         101.6869,
        "altitude":    62,
        "climate":     "Equatorial",
        "region":      "Global"
    },
    {
        "name":        "nairobi",
        "display":     "Nairobi, Kenya",
        "lat":         -1.2921,
        "lon":         36.8219,
        "altitude":    1795,
        "climate":     "Tropical Highland",
        "region":      "Global"
    },

    # ── GLOBAL: ARID/DESERT EXTREME ───────────────────────────────────
    {
        "name":        "atacama",
        "display":     "Atacama, Chile",
        "lat":         -23.4425,
        "lon":         -68.1978,
        "altitude":    2407,
        "climate":     "Hyperarid Desert",
        "region":      "Global"
    },
    {
        "name":        "dubai",
        "display":     "Dubai, UAE",
        "lat":         25.2048,
        "lon":         55.2708,
        "altitude":    5,
        "climate":     "Coastal Hot Desert",
        "region":      "Global"
    },

    # ── GLOBAL: CONTINENTAL ───────────────────────────────────────────
    {
        "name":        "berlin",
        "display":     "Berlin, Germany",
        "lat":         52.5200,
        "lon":         13.4050,
        "altitude":    34,
        "climate":     "Temperate Continental",
        "region":      "Global"
    },
    {
        "name":        "beijing",
        "display":     "Beijing, China",
        "lat":         39.9042,
        "lon":         116.4074,
        "altitude":    43,
        "climate":     "Continental + Aerosol",
        "region":      "Global"
    },
    {
        "name":        "johannesburg",
        "display":     "Johannesburg, South Africa",
        "lat":         -26.2041,
        "lon":         28.0473,
        "altitude":    1753,
        "climate":     "Subtropical Highland",
        "region":      "Global"
    },
]


# ═══════════════════════════════════════════
# GLOBAL SETTINGS
# ═══════════════════════════════════════════

START_YEAR = 2016
END_YEAR   = 2024
OUTPUT_DIR = 'solar_data'


# ═══════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════

def true_solar_time_offset(lon):
    return lon / 15.0


def utc_to_true_solar_time(df, lon, col='timestamp'):
    offset  = pd.Timedelta(hours=true_solar_time_offset(lon))
    df      = df.copy()
    df[col] = df[col] + offset
    df[col] = df[col].dt.tz_localize(None)
    return df


def get_openmeteo_client():
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


# ═══════════════════════════════════════════
# CAMS RADIATION
# ═══════════════════════════════════════════

def parse_cams_csv(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('# Observation period'):
                header_idx = i
                break
        if header_idx is None:
            return None
        header_line = lines[header_idx].strip().lstrip('# ')
        columns     = [col.strip() for col in header_line.split(';')]
        data_str    = ''.join(lines[header_idx + 1:])
        df = pd.read_csv(StringIO(data_str), sep=';', header=None, names=columns)
        df['timestamp'] = pd.to_datetime(
            df['Observation period'].str.split('/').str[1].str.replace('.0','',regex=False),
            format='%Y-%m-%dT%H:%M:%S'
        )
        df = df.rename(columns={
            'GHI': 'ghi_cams', 'DHI': 'dhi_cams', 'BNI': 'dni_cams',
            'BHI': 'bhi_cams', 'Clear sky GHI': 'clear_sky_ghi_cams',
            'Clear sky BNI': 'clear_sky_dni_cams', 'Clear sky DHI': 'clear_sky_dhi_cams',
            'TOA': 'toa_irradiance', 'Reliability': 'cams_reliability'
        })
        df = df[df['cams_reliability'] >= 0.8].copy()
        keep = ['timestamp','ghi_cams','dhi_cams','dni_cams','bhi_cams',
                'clear_sky_ghi_cams','clear_sky_dni_cams','clear_sky_dhi_cams','toa_irradiance']
        return df[keep].reset_index(drop=True)
    except Exception as e:
        print(f'    CAMS parse error: {e}')
        return None


def fetch_cams(lat, lon, location_name, start_year, end_year, output_dir):
    os.makedirs(f'{output_dir}/cams_raw', exist_ok=True)
    client  = cdsapi.Client()
    dataset = "cams-solar-radiation-timeseries"
    all_dfs = []
    for year in range(start_year, end_year + 1):
        raw_file = f'{output_dir}/cams_raw/cams_{location_name}_{year}.csv'
        if not os.path.exists(raw_file):
            request = {
                "sky_type": "observed_cloud",
                "location": {"longitude": lon, "latitude": lat},
                "altitude": ["-999"],
                "date":     [f"{year}-01-01/{year}-12-31"],
                "time_step": "1hour",
                "time_reference": "true_solar_time",
                "data_format": "csv"
            }
            try:
                client.retrieve(dataset, request).download(raw_file)
                print(f'    CAMS {year}: saved')
            except Exception as e:
                if "no data available" in str(e).lower():
                    print(f'    CAMS {year}: no data at this location — skip')
                else:
                    print(f'    CAMS {year}: error → {e}')
                continue
            time.sleep(2)
        df = parse_cams_csv(raw_file)
        if df is not None:
            all_dfs.append(df)
    if not all_dfs:
        return None
    cams_df = pd.concat(all_dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f'    CAMS total: {len(cams_df):,} rows')
    return cams_df


# ═══════════════════════════════════════════
# OPENMETEO WEATHER
# ═══════════════════════════════════════════

def fetch_openmeteo_weather(lat, lon, location_name, start_year, end_year, output_dir):
    os.makedirs(f'{output_dir}/openmeteo_raw', exist_ok=True)
    raw_file = f'{output_dir}/openmeteo_raw/weather_{location_name}.csv'
    if os.path.exists(raw_file):
        df = pd.read_csv(raw_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f'    Weather: loaded {len(df):,} rows')
        return df
    openmeteo = get_openmeteo_client()
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{start_year}-01-01",
        "end_date":   f"{end_year}-12-31",
        "hourly": [
            "temperature_2m","relative_humidity_2m","dewpoint_2m",
            "wind_speed_10m","wind_direction_10m","surface_pressure",
            "cloud_cover","cloud_cover_low","cloud_cover_mid","cloud_cover_high",
            "total_column_integrated_water_vapour","precipitation",
            "shortwave_radiation","direct_normal_irradiance","diffuse_radiation",
            "sunshine_duration","is_day"
        ],
        "wind_speed_unit": "ms",
        "timezone": "UTC"
    }
    responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
    response  = responses[0]
    hourly    = response.Hourly()
    timestamps = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"
    )
    col_names = [
        "temperature_2m","relative_humidity_2m","dewpoint_2m",
        "wind_speed_10m","wind_direction_10m","surface_pressure",
        "cloud_cover","cloud_cover_low","cloud_cover_mid","cloud_cover_high",
        "total_column_integrated_water_vapour","precipitation",
        "ghi_openmeteo","dni_openmeteo","dhi_openmeteo","sunshine_duration","is_day"
    ]
    data = {"timestamp": timestamps}
    for i, name in enumerate(col_names):
        data[name] = hourly.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(data)
    df = utc_to_true_solar_time(df, lon)
    df.to_csv(raw_file, index=False)
    print(f'    Weather: {len(df):,} rows saved')
    return df


# ═══════════════════════════════════════════
# OPENMETEO AIR QUALITY
# ═══════════════════════════════════════════

def fetch_openmeteo_airquality(lat, lon, location_name, start_year, end_year, output_dir):
    os.makedirs(f'{output_dir}/openmeteo_raw', exist_ok=True)
    raw_file = f'{output_dir}/openmeteo_raw/airquality_{location_name}.csv'
    if os.path.exists(raw_file):
        df = pd.read_csv(raw_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f'    AQ: loaded {len(df):,} rows')
        return df
    aq_start  = max(start_year, 2013)
    openmeteo = get_openmeteo_client()
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{aq_start}-01-01",
        "end_date":   f"{end_year}-12-31",
        "hourly": ["aerosol_optical_depth","dust","pm10","pm2_5","uv_index"],
        "timezone": "UTC"
    }
    responses = openmeteo.weather_api("https://air-quality-api.open-meteo.com/v1/air-quality", params=params)
    response  = responses[0]
    hourly    = response.Hourly()
    timestamps = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left"
    )
    col_names = ["aerosol_optical_depth","dust","pm10","pm2_5","uv_index"]
    data = {"timestamp": timestamps}
    for i, name in enumerate(col_names):
        data[name] = hourly.Variables(i).ValuesAsNumpy()
    df = pd.DataFrame(data)
    df = utc_to_true_solar_time(df, lon)
    df.to_csv(raw_file, index=False)
    print(f'    AQ: {len(df):,} rows saved')
    return df


# ═══════════════════════════════════════════
# AEROSOL MISSING DATA HANDLER
# ═══════════════════════════════════════════

def handle_missing_aerosol(df):
    aerosol_cols = ['aerosol_optical_depth','dust','pm10','pm2_5','uv_index']
    df = df.copy()
    df['aerosol_data_real'] = df['aerosol_optical_depth'].notna().astype(int)
    est_count = (df['aerosol_data_real'] == 0).sum()
    if est_count == 0:
        return df
    df['_month'] = df['timestamp'].dt.month
    df['_hour']  = df['timestamp'].dt.hour
    for col in aerosol_cols:
        if col not in df.columns:
            continue
        missing_mask = df[col].isna()
        if missing_mask.sum() == 0:
            continue
        clim = df[df[col].notna()].groupby(['_month','_hour'])[col].mean()
        if not clim.empty:
            for idx in df[missing_mask].index:
                key = (df.at[idx,'_month'], df.at[idx,'_hour'])
                if key in clim.index:
                    df.at[idx, col] = clim[key]
    df = df.drop(columns=['_month','_hour'])
    defaults = {'aerosol_optical_depth':0.15,'dust':0.0,'pm10':20.0,'pm2_5':10.0,'uv_index':0.0}
    for col, default in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    return df


# ═══════════════════════════════════════════
# pvlib SOLAR GEOMETRY
# ═══════════════════════════════════════════

def compute_pvlib_features(df, lat, lon, altitude):
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
    return df


# ═══════════════════════════════════════════
# ENGINEERED FEATURES
# ═══════════════════════════════════════════

def compute_days_since_rain(precip_series, threshold=5.0):
    result = []
    count  = 0
    for p in precip_series:
        count = 0 if p >= threshold else count + 1
        result.append(count / 24.0)
    return result


def engineer_features(df, lat, lon, altitude, climate, region):
    df   = df.copy()
    hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    doy  = df['timestamp'].dt.dayofyear
    df['hour_sin']  = np.sin(2 * np.pi * hour / 24)
    df['hour_cos']  = np.cos(2 * np.pi * hour / 24)
    df['day_sin']   = np.sin(2 * np.pi * doy  / 365.25)
    df['day_cos']   = np.cos(2 * np.pi * doy  / 365.25)
    df['hour']      = df['timestamp'].dt.hour
    df['month']     = df['timestamp'].dt.month
    df['year']      = df['timestamp'].dt.year
    df['kt']        = (df['ghi_openmeteo'] / df['clear_sky_ghi'].replace(0, np.nan)).clip(0,1.5).fillna(0)
    df['ghi_clear_weighted'] = df['clear_sky_ghi'] * df['cos_zenith']
    df['csi']       = (df['ghi_cams'] / df['clear_sky_ghi'].replace(0, np.nan)).clip(0,1.5).fillna(0)
    df['days_since_rain_5mm'] = compute_days_since_rain(df['precipitation'])
    # Location identity features (needed for TFT multi-location training)
    df['latitude']  = lat
    df['longitude'] = lon
    df['altitude']  = altitude
    df['climate_zone'] = climate
    df['region']    = region
    return df


# ═══════════════════════════════════════════
# MERGE
# ═══════════════════════════════════════════

def merge_all_sources(cams_df, weather_df, aq_df):
    if cams_df is None:
        return None
    for df in [cams_df, weather_df, aq_df]:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.round('h')
    merged = cams_df.merge(weather_df, on='timestamp', how='inner')
    merged = merged.merge(aq_df, on='timestamp', how='left')
    return merged


# ═══════════════════════════════════════════
# SINGLE LOCATION PIPELINE
# ═══════════════════════════════════════════

def collect_location(loc, start_year, end_year, output_dir):
    """
    Run the full data collection pipeline for one location.
    Returns True if successful, False if failed.
    """
    name     = loc['name']
    lat      = loc['lat']
    lon      = loc['lon']
    altitude = loc['altitude']
    climate  = loc['climate']
    region   = loc['region']

    final_file = f'{output_dir}/{name}_complete_dataset.csv'

    if os.path.exists(final_file):
        existing = pd.read_csv(final_file, nrows=1)
        print(f'  Already complete — skipping')
        return True

    os.makedirs(output_dir, exist_ok=True)

    try:
        # CAMS
        cams_df = fetch_cams(lat, lon, name, start_year, end_year, output_dir)
        if cams_df is None:
            print(f'  FAILED: No CAMS data — location may be outside coverage')
            return False

        # OpenMeteo Weather
        weather_df = fetch_openmeteo_weather(lat, lon, name, start_year, end_year, output_dir)

        # OpenMeteo Air Quality
        aq_df = fetch_openmeteo_airquality(lat, lon, name, start_year, end_year, output_dir)

        # Merge
        merged = merge_all_sources(cams_df, weather_df, aq_df)
        if merged is None:
            print(f'  FAILED: Merge returned None')
            return False

        # Fill missing aerosol
        merged = handle_missing_aerosol(merged)

        # pvlib geometry
        merged = compute_pvlib_features(merged, lat, lon, altitude)

        # Engineered features
        merged = engineer_features(merged, lat, lon, altitude, climate, region)

        # Save
        merged.to_csv(final_file, index=False)
        print(f'  SAVED: {final_file} — {len(merged):,} rows, {merged.shape[1]} columns')
        return True

    except Exception as e:
        print(f'  ERROR: {e}')
        return False


# ═══════════════════════════════════════════
# MAIN — COLLECT ALL 35 LOCATIONS
# ═══════════════════════════════════════════

def collect_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total     = len(LOCATIONS)
    success   = []
    failed    = []
    skipped   = []

    print(f'\nSmart Solar Grid — Multi-Location Data Collection')
    print(f'Total locations: {total}')
    print(f'Period: {START_YEAR} — {END_YEAR}')
    print(f'Output: {OUTPUT_DIR}/')
    print('=' * 60)

    for i, loc in enumerate(LOCATIONS, 1):
        print(f'\n[{i:02d}/{total}] {loc["display"]} ({loc["climate"]}, {loc["region"]})')
        final_file = f'{OUTPUT_DIR}/{loc["name"]}_complete_dataset.csv'

        if os.path.exists(final_file):
            print(f'  Already complete — skipping')
            skipped.append(loc['name'])
            continue

        result = collect_location(loc, START_YEAR, END_YEAR, OUTPUT_DIR)

        if result:
            success.append(loc['name'])
        else:
            failed.append(loc['name'])

        # Pause between locations to be polite to APIs
        if i < total:
            print(f'  Waiting 5 seconds before next location...')
            time.sleep(5)

    # Final summary
    print('\n' + '=' * 60)
    print('COLLECTION COMPLETE')
    print(f'  Successful:  {len(success) + len(skipped):02d} / {total}')
    print(f'    New:       {len(success):02d}')
    print(f'    Skipped:   {len(skipped):02d} (already existed)')
    print(f'  Failed:      {len(failed):02d}')

    if failed:
        print(f'\nFailed locations (check CAMS coverage or API errors):')
        for name in failed:
            print(f'  - {name}')

    # Save collection status report
    report_rows = []
    for loc in LOCATIONS:
        final_file = f'{OUTPUT_DIR}/{loc["name"]}_complete_dataset.csv'
        if os.path.exists(final_file):
            df   = pd.read_csv(final_file, nrows=1)
            rows = sum(1 for _ in open(final_file)) - 1
            status = 'complete'
        else:
            rows   = 0
            status = 'failed'
        report_rows.append({
            'location':   loc['name'],
            'display':    loc['display'],
            'region':     loc['region'],
            'climate':    loc['climate'],
            'lat':        loc['lat'],
            'lon':        loc['lon'],
            'status':     status,
            'rows':       rows
        })

    report = pd.DataFrame(report_rows)
    report.to_csv(f'{OUTPUT_DIR}/collection_status.csv', index=False)
    print(f'\nStatus report saved: {OUTPUT_DIR}/collection_status.csv')
    print(report[['display','region','status','rows']].to_string(index=False))

    return report


if __name__ == '__main__':
    collect_all()