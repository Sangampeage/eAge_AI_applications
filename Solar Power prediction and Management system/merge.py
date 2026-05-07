"""
Solar Data Merge Pipeline
Merges: CAMS radiation CSVs + OpenMeteo weather CSV + Air Quality CSV
Output: Single clean CSV with aligned hourly timestamps (True Solar Time)
"""

import os
import glob
import pandas as pd
import numpy as np
from io import StringIO


# ─────────────────────────────────────────────────────
# CONFIGURE PATHS — edit these
# ─────────────────────────────────────────────────────
CAMS_DIR      = "solar_data/cams_raw"          # folder with cams_bengaluru_YYYY.csv files
WEATHER_CSV   = "solar_data/openmeteo_raw/weather_bengaluru.csv"
AQ_CSV        = "solar_data/openmeteo_raw/airquality_bengaluru.csv"
OUTPUT_CSV    = "solar_data/bengaluru_merged.csv"
# ─────────────────────────────────────────────────────


# ═══════════════════════════════════════════
# STEP 1: LOAD AND PARSE CAMS FILES
# ═══════════════════════════════════════════

def parse_cams_csv(filepath):
    """
    Parse one CAMS CSV file.
    Header line starts with '# Observation period'.
    Timestamps are already in True Solar Time.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find the column header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("# Observation period"):
            header_idx = i
            break

    if header_idx is None:
        print(f"  [SKIP] No header found: {filepath}")
        return None

    # Strip leading '# ' from header
    header_line = lines[header_idx].strip().lstrip("# ")
    columns = [c.strip() for c in header_line.split(";")]

    data_str = "".join(lines[header_idx + 1:])
    df = pd.read_csv(StringIO(data_str), sep=";", header=None, names=columns)

    # Parse timestamp — use END of each interval
    # Format: 2021-01-01T00:00:00.0/2021-01-01T01:00:00.0
    df["timestamp"] = pd.to_datetime(
        df["Observation period"].str.split("/").str[1].str.replace(".0", "", regex=False),
        format="%Y-%m-%dT%H:%M:%S"
    )

    # Drop low-reliability rows
    df = df[df["Reliability"] >= 0.8].copy()

    # Rename to clean names
    df = df.rename(columns={
        "GHI":           "ghi_cams",
        "DHI":           "dhi_cams",
        "BNI":           "dni_cams",
        "BHI":           "bhi_cams",
        "Clear sky GHI": "clear_sky_ghi_cams",
        "Clear sky BNI": "clear_sky_dni_cams",
        "Clear sky DHI": "clear_sky_dhi_cams",
        "TOA":           "toa_irradiance",
        "Reliability":   "cams_reliability",
    })

    keep = [
        "timestamp",
        "ghi_cams", "dhi_cams", "dni_cams", "bhi_cams",
        "clear_sky_ghi_cams", "clear_sky_dni_cams", "clear_sky_dhi_cams",
        "toa_irradiance", "cams_reliability",
    ]
    # Only keep columns that exist (some years may differ)
    keep = [c for c in keep if c in df.columns]
    return df[keep].reset_index(drop=True)


def load_all_cams(cams_dir):
    files = sorted(glob.glob(os.path.join(cams_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CAMS CSV files found in: {cams_dir}")

    print(f"[CAMS] Found {len(files)} files")
    dfs = []
    for f in files:
        df = parse_cams_csv(f)
        if df is not None:
            dfs.append(df)
            print(f"  {os.path.basename(f)}: {len(df):,} rows")
        else:
            print(f"  {os.path.basename(f)}: FAILED — skipped")

    if not dfs:
        raise ValueError("No CAMS data parsed successfully.")

    cams = pd.concat(dfs, ignore_index=True)
    cams["timestamp"] = pd.to_datetime(cams["timestamp"]).dt.round("h")

    # Drop duplicates that can appear at year boundaries
    cams = cams.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f"[CAMS] Total: {len(cams):,} rows | {cams['timestamp'].min()} → {cams['timestamp'].max()}")
    return cams


# ═══════════════════════════════════════════
# STEP 2: LOAD OPENMETEO WEATHER
# ═══════════════════════════════════════════

def load_weather(filepath):
    print(f"[Weather] Loading: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.round("h")
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f"[Weather] {len(df):,} rows | {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ═══════════════════════════════════════════
# STEP 3: LOAD AIR QUALITY
# ═══════════════════════════════════════════

def load_airquality(filepath):
    """
    AQ timestamps come with sub-minute offsets (e.g. 05:10:32.592).
    Round to nearest hour before merging.
    Sparse NaN data for early years is expected — left join will handle it.
    """
    print(f"[AQ] Loading: {filepath}")
    df = pd.read_csv(filepath, parse_dates=["timestamp"])

    # Round the off-beat timestamps (e.g. :10:32 → :00:00)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.round("h")

    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    total = len(df)
    filled = df.dropna(subset=df.columns.difference(["timestamp"]), how="all")
    print(f"[AQ] {total:,} rows | Data present in {len(filled):,} rows | "
          f"{df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ═══════════════════════════════════════════
# STEP 4: MERGE
# ═══════════════════════════════════════════

def merge(cams_df, weather_df, aq_df):
    """
    Inner join CAMS + Weather (both must be present).
    Left join AQ (NaN where no data — acceptable).
    """
    print("\n[Merge] Joining sources on timestamp...")

    merged = cams_df.merge(weather_df, on="timestamp", how="inner")
    print(f"  After CAMS ∩ Weather: {len(merged):,} rows")

    merged = merged.merge(aq_df, on="timestamp", how="left")
    print(f"  After left-join AQ:   {len(merged):,} rows")

    # Sanity: no duplicate timestamps
    dupes = merged.duplicated(subset="timestamp").sum()
    if dupes > 0:
        print(f"  WARNING: {dupes} duplicate timestamps found — keeping first occurrence")
        merged = merged.drop_duplicates(subset="timestamp")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


# ═══════════════════════════════════════════
# STEP 5: QUALITY REPORT
# ═══════════════════════════════════════════

def quality_report(df):
    print("\n" + "=" * 55)
    print("MERGE QUALITY REPORT")
    print("=" * 55)
    print(f"  Rows:        {len(df):,}")
    print(f"  Columns:     {df.shape[1]}")
    print(f"  Date range:  {df['timestamp'].min()}  →  {df['timestamp'].max()}")
    print()

    # Missing value summary — only show columns with any nulls
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print("  No missing values.")
    else:
        print("  Missing values per column:")
        for col, count in null_counts.items():
            pct = 100 * count / len(df)
            print(f"    {col:<45} {count:>7,}  ({pct:.1f}%)")

    print()
    print("  Sample (first 3 daytime rows):")
    sample = df[df["ghi_cams"] > 0].head(3)
    print(sample[["timestamp", "ghi_cams", "ghi_openmeteo", "temperature_2m", "aerosol_optical_depth"]].to_string(index=False))
    print("=" * 55)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def run():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    if os.path.exists(OUTPUT_CSV):
        print(f"Output already exists: {OUTPUT_CSV}")
        print("Delete it to rerun. Loading existing file...")
        df = pd.read_csv(OUTPUT_CSV, parse_dates=["timestamp"])
        quality_report(df)
        return df

    cams_df    = load_all_cams(CAMS_DIR)
    weather_df = load_weather(WEATHER_CSV)
    aq_df      = load_airquality(AQ_CSV)

    merged = merge(cams_df, weather_df, aq_df)
    quality_report(merged)

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  |  Shape: {merged.shape}")
    return merged


if __name__ == "__main__":
    df = run()