"""
Smart Solar Grid — ML Pipeline with MLflow
GHI/CSI Prediction for Bengaluru
Trains: XGBoost, LightGBM, RandomForest, ExtraTrees, SVR, Stacking Ensemble
Tracks: All experiments, metrics, features, artifacts in MLflow
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os
import json
import joblib
from datetime import datetime

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────
DATA_PATH      = 'solar_data/bengaluru_complete_dataset.csv'
OUTPUT_DIR     = 'ml_outputs'
MLFLOW_URI     = 'mlruns'
EXPERIMENT     = 'GHI_Prediction_Bengaluru_CSI'
TARGET         = 'ghi_cams'   # raw W/m² — more stable than CSI for R²
# CSI = ghi_cams / clear_sky_ghi — high variance at low elevation
# Train on ghi_cams directly, cleaner R² signal
TRAIN_END      = '2022-12-31'
VAL_END        = '2023-12-31'
LAG_GAP_HOURS  = 168        # 7-day gap between splits
RANDOM_STATE   = 42
# ─────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════

def load_data(path):
    print(f'\n[Load] Reading: {path}')
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f'  Rows: {len(df):,}  |  Cols: {df.shape[1]}')
    print(f'  Range: {df["timestamp"].min()} → {df["timestamp"].max()}')
    return df


# ═══════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════

def engineer_features(df):
    """
    Complete feature engineering pipeline.
    Order matters:
      1. Compute lags on full 24h data first
      2. Compute rolling stats on full 24h data
      3. Compute physics and interaction features
      4. Add Bengaluru-specific features
      5. Filter to daytime AFTER all engineering
    """
    print('\n[Feature Engineering]')
    df = df.copy()

    # ── GROUP 1: LAG FEATURES ────────────────────────
    print('  Computing lag features...')

    # GHI lags — give model memory of past irradiance
    for lag in [1, 2, 3, 6, 24, 48, 168]:
        df[f'ghi_lag_{lag}h']   = df['ghi_openmeteo'].shift(lag)

    # Cloud lags — cloud transitions are most important
    for lag in [1, 2, 3, 6, 24]:
        df[f'cloud_lag_{lag}h'] = df['cloud_cover'].shift(lag)

    # Aerosol lags — dust storm buildup
    for lag in [1, 3, 6, 24]:
        df[f'aod_lag_{lag}h']   = df['aerosol_optical_depth'].shift(lag)

    # Temperature and humidity lags
    for lag in [1, 3, 24]:
        df[f'temp_lag_{lag}h']  = df['temperature_2m'].shift(lag)
        df[f'humid_lag_{lag}h'] = df['relative_humidity_2m'].shift(lag)

    # kt lags — clearness index history
    for lag in [1, 2, 3, 24]:
        df[f'kt_lag_{lag}h']    = df['kt'].shift(lag)

    # ── GROUP 2: ROLLING WINDOW STATISTICS ────────────
    print('  Computing rolling statistics...')

    # GHI rolling stats
    df['ghi_roll_mean_3h']   = df['ghi_openmeteo'].rolling(3,  min_periods=1).mean()
    df['ghi_roll_mean_6h']   = df['ghi_openmeteo'].rolling(6,  min_periods=1).mean()
    df['ghi_roll_mean_24h']  = df['ghi_openmeteo'].rolling(24, min_periods=1).mean()
    df['ghi_roll_max_24h']   = df['ghi_openmeteo'].rolling(24, min_periods=1).max()
    df['ghi_roll_std_3h']    = df['ghi_openmeteo'].rolling(3,  min_periods=2).std().fillna(0)
    df['ghi_roll_std_24h']   = df['ghi_openmeteo'].rolling(24, min_periods=2).std().fillna(0)

    # Cloud rolling stats
    df['cloud_roll_mean_3h']  = df['cloud_cover'].rolling(3,  min_periods=1).mean()
    df['cloud_roll_mean_6h']  = df['cloud_cover'].rolling(6,  min_periods=1).mean()
    df['cloud_roll_max_24h']  = df['cloud_cover'].rolling(24, min_periods=1).max()

    # kt rolling stats — clearness trend
    df['kt_roll_mean_3h']     = df['kt'].rolling(3,  min_periods=1).mean()
    df['kt_roll_std_3h']      = df['kt'].rolling(3,  min_periods=2).std().fillna(0)
    df['kt_roll_mean_24h']    = df['kt'].rolling(24, min_periods=1).mean()

    # Precipitation rolling sums — dust cleaning signal
    df['precip_sum_6h']       = df['precipitation'].rolling(6,  min_periods=1).sum()
    df['precip_sum_24h']      = df['precipitation'].rolling(24, min_periods=1).sum()
    df['precip_sum_72h']      = df['precipitation'].rolling(72, min_periods=1).sum()

    # Aerosol rolling
    df['aod_roll_mean_24h']   = df['aerosol_optical_depth'].rolling(24, min_periods=1).mean()
    df['aod_roll_max_24h']    = df['aerosol_optical_depth'].rolling(24, min_periods=1).max()

    # ── GROUP 3: PHYSICS-DERIVED FEATURES ─────────────
    print('  Computing physics features...')

    ghi_safe = df['ghi_openmeteo'].clip(lower=1)

    # Beam and diffuse fractions — sky clarity indicators
    df['beam_fraction']     = df['dni_openmeteo'] / ghi_safe
    df['diffuse_fraction']  = df['dhi_openmeteo'] / ghi_safe
    df['beam_fraction']     = df['beam_fraction'].clip(0, 2)
    df['diffuse_fraction']  = df['diffuse_fraction'].clip(0, 2)

    # POA proxy — approximate panel irradiance
    df['poa_proxy']         = df['ghi_openmeteo'] * df['cos_zenith']

    # Optical air mass — atmosphere thickness at current zenith
    zenith_rad              = np.radians(df['solar_zenith'].clip(0, 89.9))
    df['air_mass']          = 1 / (np.cos(zenith_rad) + 0.001)
    df['air_mass']          = df['air_mass'].clip(1, 40)

    # Linke turbidity proxy — atmosphere dirtiness
    df['turbidity_proxy']   = df['aerosol_optical_depth'] * df['air_mass']

    # Normalized clearness change — rate of sky change
    df['kt_delta_1h']       = df['kt'] - df['kt_lag_1h']
    df['cloud_delta_1h']    = df['cloud_cover'] - df['cloud_lag_1h']

    # sunshine fraction — direct sky clarity measurement
    # 3600 = full clear hour, 0 = completely overcast
    # This is stronger than cloud_cover because it measures actual transmission
    df['sunshine_fraction']    = df['sunshine_duration'] / 3600.0
    df['sunshine_lag_1h']      = df['sunshine_duration'].shift(1)
    df['sunshine_lag_3h']      = df['sunshine_duration'].shift(3)
    df['sunshine_roll_mean_3h']= df['sunshine_duration'].rolling(3, min_periods=1).mean()
    df['sunshine_roll_mean_6h']= df['sunshine_duration'].rolling(6, min_periods=1).mean()

    # ── GROUP 4: ATMOSPHERIC INTERACTIONS ─────────────
    print('  Computing atmospheric interactions...')

    # Combined sky opacity
    df['humid_cloud']       = df['relative_humidity_2m'] * df['cloud_cover'] / 100

    # Net dust effect (rain washes dust)
    df['dust_loading']      = df['aerosol_optical_depth'] * (
                                1 - df['precip_sum_24h'].clip(0, 50) / 50
                              )

    # Wind dust clearing
    df['wind_dust']         = df['wind_speed_10m'] * df['aerosol_optical_depth']

    # Temperature departure from monthly climatology
    monthly_mean_temp       = df.groupby('month')['temperature_2m'].transform('mean')
    df['temp_departure']    = df['temperature_2m'] - monthly_mean_temp

    # Surface wetness proxy
    df['surface_wet']       = df['precip_sum_24h'] / (df['precip_sum_24h'] + 1)

    # Combined extinction
    df['aerosol_cloud']     = df['aerosol_optical_depth'] * df['cloud_cover'] / 100

    # Zenith × clearness — single strongest physics feature
    df['zenith_kt']         = df['cos_zenith'] * df['kt']
    df['zenith_cloud']      = df['cos_zenith'] * (1 - df['cloud_cover'] / 100)
    df['zenith_cloud']      = df['zenith_cloud'].clip(0, 1)

    # ── GROUP 5: BENGALURU-SPECIFIC FEATURES ──────────
    print('  Computing location-specific features...')

    # Monsoon season flag (Bengaluru: Jun-Sep is SW monsoon)
    df['is_monsoon']        = df['month'].isin([6, 7, 8, 9]).astype(int)

    # NE monsoon (Oct-Nov for Tamil Nadu / South Karnataka)
    df['is_ne_monsoon']     = df['month'].isin([10, 11]).astype(int)

    # Season encoding (India-specific)
    def get_season(month):
        if month in [12, 1, 2]:  return 0  # winter
        if month in [3, 4, 5]:   return 1  # pre-summer
        if month in [6, 7, 8, 9]:return 2  # SW monsoon
        return 3                             # post-monsoon

    df['season']            = df['month'].apply(get_season)

    # Morning vs afternoon (cloud formation patterns differ)
    df['is_morning']        = (df['hour'] < 12).astype(int)

    # Solar noon distance (simpler than sin/cos for trees)
    df['solar_noon_dist']   = abs(df['hour'] - 12)

    # ── FILTER TO DAYTIME ONLY ────────────────────────
    print('  Filtering to daytime rows...')
    before = len(df)
    df = df[df['is_day_pvlib'] == 1].copy()
    print(f'  Rows: {before:,} → {len(df):,} (daytime only, elevation > 0 deg)')

    return df


# ═══════════════════════════════════════════
# STEP 3 — FEATURE SELECTION
# ═══════════════════════════════════════════

def get_feature_sets(df):
    """
    Returns 6 progressively richer feature sets
    for ablation study experiments.
    """

    # Features to never include (leakage or unavailable)
    EXCLUDE = [
        'timestamp', 'ghi_cams', 'dhi_cams', 'dni_cams', 'bhi_cams',
        'clear_sky_ghi_cams', 'clear_sky_dni_cams', 'clear_sky_dhi_cams',
        'toa_irradiance', 'cams_reliability',
        'csi',           # this is the target
        'is_day',        # duplicate of is_day_pvlib
        'solar_elevation',# duplicate of zenith
        'year',          # causes overfitting
        'clear_sky_ghi_cams', 'radiation_source',
        'climate_zone', 'region',
        'latitude', 'longitude', 'altitude_m',
        'uv_index',         # leakage risk (correlates with GHI directly)
        'aerosol_data_real', # meta-column, not a physical signal
        'sunshine_duration'  # excluded — use sunshine_fraction (engineered) insteadeta-column
        'sunshine_duration'  # use sunshine_fraction and rolling instead
    ]

    all_cols = [c for c in df.columns if c not in EXCLUDE]

    # Base features — no lags, no rolling
    lag_cols     = [c for c in all_cols if 'lag'  in c]
    roll_cols    = [c for c in all_cols if 'roll' in c or 'sum' in c]
    phys_cols    = ['beam_fraction','diffuse_fraction','poa_proxy',
                    'air_mass','turbidity_proxy','kt_delta_1h','cloud_delta_1h']
    inter_cols   = ['humid_cloud','dust_loading','wind_dust','temp_departure',
                    'surface_wet','aerosol_cloud','zenith_kt','zenith_cloud']
    india_cols   = ['is_monsoon','is_ne_monsoon','season','is_morning','solar_noon_dist']

    base_cols = [c for c in all_cols
                 if c not in lag_cols + roll_cols + phys_cols + inter_cols + india_cols]

    feature_sets = {
        'v1_base':         base_cols,
        'v2_lags':         base_cols + lag_cols,
        'v3_rolling':      base_cols + lag_cols + roll_cols,
        'v4_physics':      base_cols + lag_cols + roll_cols + phys_cols,
        'v5_interactions': base_cols + lag_cols + roll_cols + phys_cols + inter_cols,
        'v6_full':         base_cols + lag_cols + roll_cols + phys_cols + inter_cols + india_cols,
    }

    # Validate all columns exist in df
    valid_sets = {}
    for name, cols in feature_sets.items():
        valid = [c for c in cols if c in df.columns]
        valid_sets[name] = valid
        print(f'  Feature set {name}: {len(valid)} features')

    return valid_sets


# ═══════════════════════════════════════════
# STEP 4 — TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════

def split_data(df, feature_cols, target=TARGET):
    """
    Time-based split with 168-hour gap to prevent
    lag feature leakage across boundaries.
    """
    train = df[df['timestamp'] <= TRAIN_END].copy()
    val   = df[(df['timestamp'] > TRAIN_END) & (df['timestamp'] <= VAL_END)].copy()
    test  = df[df['timestamp'] > VAL_END].copy()

    # Apply temporal gap — drop first LAG_GAP_HOURS rows of val and test
    val  = val.iloc[LAG_GAP_HOURS:].copy()
    test = test.iloc[LAG_GAP_HOURS:].copy()

    # Drop rows with NaN from lag computation
    all_split = pd.concat([train, val, test])
    for split_df in [train, val, test]:
        split_df.dropna(subset=feature_cols + [target], inplace=True)

    X_train = train[feature_cols].values
    y_train = train[target].values
    X_val   = val[feature_cols].values
    y_val   = val[target].values
    X_test  = test[feature_cols].values
    y_test  = test[target].values

    print(f'\n  Train: {len(X_train):,} rows ({TRAIN_END})')
    print(f'  Val:   {len(X_val):,} rows')
    print(f'  Test:  {len(X_test):,} rows ({VAL_END} onwards)')

    return X_train, y_train, X_val, y_val, X_test, y_test, train, val, test


# ═══════════════════════════════════════════
# STEP 5 — METRICS
# ═══════════════════════════════════════════

def compute_metrics(y_true, y_pred, prefix='val'):
    """Compute all evaluation metrics."""
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mbe  = np.mean(y_pred - y_true)   # positive = overpredict

    # Directional accuracy — does prediction move in correct direction?
    if len(y_true) > 1:
        actual_delta = np.diff(y_true)
        pred_delta   = np.diff(y_pred)
        direction_acc = np.mean(np.sign(actual_delta) == np.sign(pred_delta))
    else:
        direction_acc = 0.0

    return {
        f'{prefix}_r2':           round(float(r2),   4),
        f'{prefix}_mae':          round(float(mae),  4),
        f'{prefix}_rmse':         round(float(rmse), 4),
        f'{prefix}_mbe':          round(float(mbe),  4),
        f'{prefix}_direction_acc':round(float(direction_acc), 4),
    }


def compute_segment_metrics(df_split, y_pred, target=TARGET):
    """Compute metrics by time segment for deeper analysis."""
    df = df_split.copy()
    df['pred'] = y_pred
    metrics = {}

    # By hour group
    df['hour_group'] = pd.cut(df['hour'], bins=[5,9,12,15,18,20],
                               labels=['6-9am','9am-12pm','12-3pm','3-6pm','6-8pm'])
    for grp, sub in df.dropna(subset=['hour_group']).groupby('hour_group', observed=True):
        if len(sub) > 10:
            r2 = r2_score(sub[target], sub['pred'])
            metrics[f'r2_hour_{grp}'] = round(float(r2), 4)

    # By season
    for s, name in [(0,'winter'),(1,'pre_summer'),(2,'monsoon'),(3,'post_monsoon')]:
        sub = df[df['season'] == s] if 'season' in df.columns else pd.DataFrame()
        if len(sub) > 10:
            r2 = r2_score(sub[target], sub['pred'])
            metrics[f'r2_season_{name}'] = round(float(r2), 4)

    # Clear days vs cloudy days
    if 'cloud_cover' in df.columns:
        clear = df[df['cloud_cover'] < 20]
        cloudy = df[df['cloud_cover'] > 70]
        if len(clear) > 10:
            metrics['r2_clear_days'] = round(float(r2_score(clear[target], clear['pred'])), 4)
        if len(cloudy) > 10:
            metrics['r2_cloudy_days'] = round(float(r2_score(cloudy[target], cloudy['pred'])), 4)

    return metrics


# ═══════════════════════════════════════════
# STEP 6 — ARTIFACT PLOTS
# ═══════════════════════════════════════════

def plot_predicted_vs_actual(y_true, y_pred, model_name, feature_set):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} | {feature_set}', fontsize=13, fontweight='bold')

    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=5, color='steelblue')
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
    r2 = r2_score(y_true, y_pred)
    axes[0].set_xlabel('Actual CSI')
    axes[0].set_ylabel('Predicted CSI')
    axes[0].set_title(f'Predicted vs Actual (R²={r2:.4f})')
    axes[0].grid(alpha=0.3)

    # Residuals distribution
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].axvline(residuals.mean(), color='orange', linestyle='--',
                    label=f'Mean={residuals.mean():.4f}')
    axes[1].set_xlabel('Residual (Predicted - Actual)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/pred_vs_actual_{model_name}_{feature_set}.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def plot_residuals_by_hour(df_val, y_pred, model_name, feature_set, target=TARGET):
    df = df_val.copy()
    df['pred']     = y_pred
    df['residual'] = df['pred'] - df[target]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} | Residual Analysis', fontsize=13, fontweight='bold')

    # By hour
    hourly = df.groupby('hour')['residual'].agg(['mean','std'])
    axes[0].bar(hourly.index, hourly['mean'], color='steelblue', alpha=0.7, label='Mean residual')
    axes[0].fill_between(hourly.index,
                         hourly['mean'] - hourly['std'],
                         hourly['mean'] + hourly['std'],
                         alpha=0.3, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Mean Residual by Hour')
    axes[0].grid(alpha=0.3)

    # By month
    monthly = df.groupby('month')['residual'].agg(['mean','std'])
    months  = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
    axes[1].bar(monthly.index, monthly['mean'], color='coral', alpha=0.7)
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_xlabel('Month')
    axes[1].set_xticks(range(1,13))
    axes[1].set_xticklabels(months, rotation=45)
    axes[1].set_ylabel('Mean Residual')
    axes[1].set_title('Mean Residual by Month')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/residuals_{model_name}_{feature_set}.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


def plot_feature_importance(model, feature_cols, model_name, feature_set, top_n=30):
    """Works for tree-based models that have feature_importances_."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        return None

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_cols[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
    ax.barh(range(top_n), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{model_name} — Top {top_n} Feature Importances')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/importance_{model_name}_{feature_set}.png'
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


# ═══════════════════════════════════════════
# STEP 7 — TRAIN ONE MODEL WITH MLFLOW
# ═══════════════════════════════════════════

def train_and_log(model, model_name, model_params,
                  X_train, y_train, X_val, y_val,
                  X_test, y_test,
                  feature_cols, feature_set,
                  df_val, df_test,
                  needs_scaling=False):
    """
    Train one model, compute all metrics, log everything to MLflow.
    Returns val_r2 for comparison.
    """
    with mlflow.start_run(run_name=f'{model_name}__{feature_set}'):

        # Scale if needed (SVR)
        if needs_scaling:
            scaler  = StandardScaler()
            X_tr    = scaler.fit_transform(X_train)
            X_v     = scaler.transform(X_val)
            X_te    = scaler.transform(X_test)
        else:
            X_tr, X_v, X_te = X_train, X_val, X_test

        # Train — pass eval_set for XGBoost/LightGBM early stopping
        t0         = datetime.now()
        model_type = type(model).__name__
        if model_type == 'XGBRegressor':
            model.fit(X_tr, y_train,
                      eval_set=[(X_v, y_val)],
                      verbose=False)
        elif model_type == 'LGBMRegressor':
            import lightgbm as lgb
            model.fit(X_tr, y_train,
                      eval_set=[(X_v, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=-1)])
        else:
            model.fit(X_tr, y_train)
        train_sec = (datetime.now() - t0).total_seconds()

        # Predict
        y_pred_train = model.predict(X_tr)
        y_pred_val   = model.predict(X_v)
        y_pred_test  = model.predict(X_te)

        # Metrics
        train_metrics = compute_metrics(y_train, y_pred_train, prefix='train')
        val_metrics   = compute_metrics(y_val,   y_pred_val,   prefix='val')
        test_metrics  = compute_metrics(y_test,  y_pred_test,  prefix='test')
        seg_metrics   = compute_segment_metrics(df_val, y_pred_val)

        overfitting_gap = train_metrics['train_r2'] - val_metrics['val_r2']

        all_metrics = {
            **train_metrics,
            **val_metrics,
            **test_metrics,
            **seg_metrics,
            'overfitting_gap':    round(overfitting_gap, 4),
            'training_time_secs': round(train_sec, 2),
        }

        # Log parameters
        mlflow.log_params({
            'model_type':    model_name,
            'feature_set':   feature_set,
            'n_features':    len(feature_cols),
            'target':        TARGET,
            'train_end':     TRAIN_END,
            'val_end':       VAL_END,
            'needs_scaling': needs_scaling,
            **{f'param_{k}': v for k, v in model_params.items()}
        })

        # Log metrics
        mlflow.log_metrics(all_metrics)

        # Log artifacts
        p1 = plot_predicted_vs_actual(y_pred_val, y_val, model_name, feature_set)
        p2 = plot_residuals_by_hour(df_val, y_pred_val, model_name, feature_set)
        p3 = plot_feature_importance(model, feature_cols, model_name, feature_set)

        mlflow.log_artifact(p1)
        mlflow.log_artifact(p2)
        if p3:
            mlflow.log_artifact(p3)

        # Log feature list
        feat_path = f'{OUTPUT_DIR}/features_{model_name}_{feature_set}.json'
        with open(feat_path, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        mlflow.log_artifact(feat_path)

        # Save model
        model_path = f'{OUTPUT_DIR}/model_{model_name}_{feature_set}.pkl'
        if needs_scaling:
            joblib.dump({'model': model, 'scaler': scaler}, model_path)
        else:
            joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Print summary
        print(f'    {model_name:<20} | {feature_set:<18} | '
              f'val_r2={val_metrics["val_r2"]:.4f} | '
              f'overfit_gap={overfitting_gap:.4f} | '
              f'{train_sec:.1f}s')

    return val_metrics['val_r2']


# ═══════════════════════════════════════════
# STEP 8 — MODEL DEFINITIONS
# ═══════════════════════════════════════════

def get_models():
    """
    Returns all model configurations.
    Each entry: (model_object, model_name, params_dict, needs_scaling)
    """
    models = [
        (
            XGBRegressor(
                n_estimators=800, learning_rate=0.05, max_depth=6,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                early_stopping_rounds=50,
                eval_metric='rmse', random_state=RANDOM_STATE,
                n_jobs=-1, verbosity=0
            ),
            'XGBoost',
            {'n_estimators':800,'lr':0.05,'max_depth':6,'subsample':0.8},
            False
        ),
        (
            LGBMRegressor(
                n_estimators=800, learning_rate=0.05, num_leaves=63,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            ),
            'LightGBM',
            {'n_estimators':800,'lr':0.05,'num_leaves':63,'subsample':0.8},
            False
        ),
        (
            RandomForestRegressor(
                n_estimators=400, max_depth=25, min_samples_leaf=5,
                max_features='sqrt', oob_score=True,
                random_state=RANDOM_STATE, n_jobs=-1
            ),
            'RandomForest',
            {'n_estimators':400,'max_depth':25,'min_samples_leaf':5},
            False
        ),
        (
            ExtraTreesRegressor(
                n_estimators=400, max_depth=25, min_samples_leaf=5,
                max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1
            ),
            'ExtraTrees',
            {'n_estimators':400,'max_depth':25,'min_samples_leaf':5},
            False
        ),
        (
            SVR(kernel='rbf', C=50, epsilon=0.01, gamma='scale'),
            'SVR',
            {'kernel':'rbf','C':50,'epsilon':0.01},
            True   # needs StandardScaler
        ),
    ]
    return models


# ═══════════════════════════════════════════
# STEP 9 — ABLATION STUDY
# (Feature set comparison for best model)
# ═══════════════════════════════════════════

def run_ablation_study(df, feature_sets):
    """
    Run XGBoost across all 6 feature sets to measure
    contribution of each feature engineering group.
    """
    print('\n[Ablation Study] XGBoost across feature sets')
    mlflow.set_experiment(f'{EXPERIMENT}_ablation')
    results = {}

    for fs_name, feature_cols in feature_sets.items():
        X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = \
            split_data(df, feature_cols)

        model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30,
            eval_metric='rmse', random_state=RANDOM_STATE,
            n_jobs=-1, verbosity=0
        )

        val_r2 = train_and_log(
            model=model,
            model_name='XGBoost',
            model_params={'n_estimators':500,'lr':0.05,'max_depth':6},
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            feature_cols=feature_cols,
            feature_set=fs_name,
            df_val=val_df, df_test=test_df
        )
        results[fs_name] = val_r2

    print('\n  Ablation results:')
    for fs, r2 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f'    {fs:<20} val_r2={r2:.4f}')

    best_fs = max(results, key=results.get)
    print(f'\n  Best feature set: {best_fs} (val_r2={results[best_fs]:.4f})')
    return best_fs, results


# ═══════════════════════════════════════════
# STEP 10 — FULL MODEL COMPARISON
# (All models on best feature set)
# ═══════════════════════════════════════════

def run_model_comparison(df, feature_cols, best_fs):
    """
    Train all 5 models on the best feature set.
    Compare side by side in MLflow.
    """
    print(f'\n[Model Comparison] All models on feature set: {best_fs}')
    mlflow.set_experiment(f'{EXPERIMENT}_model_comparison')

    X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = \
        split_data(df, feature_cols)

    models  = get_models()
    results = {}

    for model, name, params, needs_scale in models:
        print(f'\n  Training {name}...')
        # train_and_log handles eval_set internally for XGBoost/LightGBM
        val_r2 = train_and_log(
            model=model, model_name=name, model_params=params,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            feature_cols=feature_cols, feature_set=best_fs,
            df_val=val_df, df_test=test_df,
            needs_scaling=needs_scale
        )
        results[name] = val_r2

    print('\n  Model comparison results:')
    for name, r2 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f'    {name:<20} val_r2={r2:.4f}')

    return results


# ═══════════════════════════════════════════
# STEP 11 — STACKING ENSEMBLE
# ═══════════════════════════════════════════

def run_stacking_ensemble(df, feature_cols, best_fs):
    """
    Stack best 3 models (XGBoost + LightGBM + ExtraTrees)
    with Ridge meta-learner.
    """
    print(f'\n[Stacking Ensemble] Building on: {best_fs}')
    mlflow.set_experiment(f'{EXPERIMENT}_model_comparison')

    X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = \
        split_data(df, feature_cols)

    base_models = [
        ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                              subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)),
        ('lgbm', LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ('et',   ExtraTreesRegressor(n_estimators=300, max_depth=25,
                                      random_state=RANDOM_STATE, n_jobs=-1)),
    ]

    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        passthrough=True,
        n_jobs=-1
    )

    stacking.fit(X_train, y_train)
    y_pred_val  = stacking.predict(X_val)
    y_pred_test = stacking.predict(X_test)

    with mlflow.start_run(run_name=f'StackingEnsemble__{best_fs}'):
        val_m   = compute_metrics(y_val,   y_pred_val,  'val')
        test_m  = compute_metrics(y_test,  y_pred_test, 'test')
        seg_m   = compute_segment_metrics(val_df, y_pred_val)
        mlflow.log_params({
            'model_type':    'StackingEnsemble',
            'feature_set':   best_fs,
            'n_features':    len(feature_cols),
            'base_models':   'XGBoost+LightGBM+ExtraTrees',
            'meta_learner':  'Ridge',
            'cv_folds':      5
        })
        mlflow.log_metrics({**val_m, **test_m, **seg_m})
        p1 = plot_predicted_vs_actual(y_pred_val, y_val, 'StackingEnsemble', best_fs)
        p2 = plot_residuals_by_hour(val_df, y_pred_val, 'StackingEnsemble', best_fs)
        mlflow.log_artifact(p1); mlflow.log_artifact(p2)
        model_path = f'{OUTPUT_DIR}/model_Stacking_{best_fs}.pkl'
        joblib.dump(stacking, model_path)
        mlflow.log_artifact(model_path)
        print(f'    StackingEnsemble      val_r2={val_m["val_r2"]:.4f}')

    return val_m['val_r2']


# ═══════════════════════════════════════════
# STEP 12 — REGISTER BEST MODEL
# ═══════════════════════════════════════════

def register_best_model(df, feature_cols, best_model_name, best_fs):
    """
    Re-train best model on train+val combined,
    evaluate on test, register in MLflow Model Registry.
    """
    print(f'\n[Register Best Model] {best_model_name} on {best_fs}')
    mlflow.set_experiment(f'{EXPERIMENT}_champion')

    # Train on train+val combined for maximum data
    trainval = df[df['timestamp'] <= VAL_END].copy()
    test_df  = df[df['timestamp'] > VAL_END].copy()
    test_df  = test_df.iloc[LAG_GAP_HOURS:].copy()

    trainval.dropna(subset=feature_cols + [TARGET], inplace=True)
    test_df.dropna(subset=feature_cols + [TARGET], inplace=True)

    X_trainval = trainval[feature_cols].values
    y_trainval = trainval[TARGET].values
    X_test     = test_df[feature_cols].values
    y_test     = test_df[TARGET].values

    # Train champion model
    champion = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    )
    champion.fit(X_trainval, y_trainval)
    y_pred = champion.predict(X_test)

    test_metrics = compute_metrics(y_test, y_pred, 'test')
    seg_metrics  = compute_segment_metrics(test_df, y_pred)

    with mlflow.start_run(run_name='Champion_Model') as run:
        mlflow.log_params({
            'model_type':      best_model_name,
            'feature_set':     best_fs,
            'n_features':      len(feature_cols),
            'trained_on':      'train+val combined',
            'evaluated_on':    'test 2024',
        })
        mlflow.log_metrics({**test_metrics, **seg_metrics})

        # Log feature list
        feat_path = f'{OUTPUT_DIR}/champion_features.json'
        with open(feat_path, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        mlflow.log_artifact(feat_path)

        # Save and register
        model_path = f'{OUTPUT_DIR}/champion_model.pkl'
        joblib.dump({'model': champion, 'feature_cols': feature_cols}, model_path)
        mlflow.log_artifact(model_path)

        # Register in MLflow Model Registry
        mlflow.sklearn.log_model(
            champion,
            artifact_path='model',
            registered_model_name='GHI_Bengaluru_Champion'
        )

        print(f'\n  Champion test metrics:')
        for k, v in test_metrics.items():
            print(f'    {k}: {v}')
        print(f'\n  Registered as: GHI_Bengaluru_Champion')
        print(f'  Run ID: {run.info.run_id}')


# ═══════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════

def run_pipeline():
    print('=' * 65)
    print('Smart Solar Grid — ML Training Pipeline with MLflow')
    print('=' * 65)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    print(f'\nMLflow tracking URI: {MLFLOW_URI}')
    print(f'Run: mlflow ui --port 5000 to view results\n')

    # Load data
    df = load_data(DATA_PATH)

    # Feature engineering
    df = engineer_features(df)

    # Get feature sets for ablation
    feature_sets = get_feature_sets(df)

    # Step A: Ablation study — find best feature set
    best_fs, ablation_results = run_ablation_study(df, feature_sets)

    # Step B: Compare all models on best feature set
    best_feature_cols = feature_sets[best_fs]
    model_results = run_model_comparison(df, best_feature_cols, best_fs)

    # Step C: Stacking ensemble
    stack_r2 = run_stacking_ensemble(df, best_feature_cols, best_fs)
    model_results['StackingEnsemble'] = stack_r2

    # Find overall best model
    best_model_name = max(model_results, key=model_results.get)
    best_r2         = model_results[best_model_name]

    print(f'\n{"=" * 65}')
    print('FINAL RESULTS SUMMARY')
    print(f'{"=" * 65}')
    print(f'Best feature set:  {best_fs}')
    print(f'Best model:        {best_model_name} (val_r2={best_r2:.4f})')
    print(f'\nAll model results:')
    for name, r2 in sorted(model_results.items(), key=lambda x: x[1], reverse=True):
        flag = ' ← CHAMPION' if name == best_model_name else ''
        print(f'  {name:<22} val_r2={r2:.4f}{flag}')

    # Step D: Register champion model
    register_best_model(df, best_feature_cols, best_model_name, best_fs)

    print(f'\nDone. Open MLflow UI:')
    print(f'  cd to your project folder')
    print(f'  mlflow ui --port 5000')
    print(f'  Open: http://localhost:5000')

    return model_results


if __name__ == '__main__':
    run_pipeline()