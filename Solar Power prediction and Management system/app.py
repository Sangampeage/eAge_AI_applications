import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Streamlit config MUST be the first command
st.set_page_config(page_title="Solar Power Prediction & Management", layout="wide")

DB_PATH = 'solar_predictions.db'

def load_predictions():
    conn = sqlite3.connect(DB_PATH)
    models = ['xgboost', 'lightgbm', 'randomforest', 'extratrees', 'svr']
    df = None
    
    try:
        for model in models:
            table_name = f'predictions_{model}'
            # Query and explicitly cast to datetime strings
            temp_df = pd.read_sql(f'SELECT timestamp, predicted_ghi AS {model} FROM {table_name}', conn)
            
            if temp_df.empty:
                continue
                
            temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
            
            # ──────── SAFETY: De-duplicate just in case ────────
            temp_df = temp_df.drop_duplicates(subset=['timestamp'])
            
            if df is None:
                df = temp_df
            else:
                df = pd.merge(df, temp_df, on='timestamp', how='outer')
                
        if df is not None and not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Standardize: No Timezones for easier filtering (treat both as local/UTC equally)
            if getattr(df['timestamp'].dtype, 'tz', None) is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            # rename columns to proper case
            df = df.rename(columns={'xgboost':'XGBoost', 'lightgbm':'LightGBM', 'randomforest':'RandomForest', 'extratrees':'ExtraTrees', 'svr':'SVR'})
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def load_sensor_data():
    conn = sqlite3.connect(DB_PATH)
    try:
        # Load hourly historical data
        df = pd.read_sql('SELECT * FROM sensor_data', conn)
        if not df.empty:
            df['hour_timestamp'] = pd.to_datetime(df['hour_timestamp'])
            # Shift back by 1 hour so 4-5pm displays as 4pm (start of hour)
            df['hour_timestamp'] = df['hour_timestamp'] - pd.Timedelta(hours=1)
            if getattr(df['hour_timestamp'].dtype, 'tz', None) is not None:
                df['hour_timestamp'] = df['hour_timestamp'].dt.tz_localize(None)
        
        # ──────── FIXED: Fetch current hour partial average ────────
        now = datetime.now()
        current_hour_str = now.replace(minute=0, second=0, microsecond=0).isoformat()
        
        # Check if we already have the previous full hour (which would be stored as next_hour per user convention)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        
        # Query raw readings for the current active hour
        raw_df = pd.read_sql("""
            SELECT ghi_wm2 FROM sensor_raw_readings 
            WHERE timestamp >= ? AND crc_valid = 1
        """, conn, params=(now.replace(minute=0, second=0, microsecond=0).isoformat(),))
        
        if not raw_df.empty:
            current_avg = raw_df['ghi_wm2'].mean()
            current_row = pd.DataFrame({
                'hour_timestamp': [now.replace(minute=0, second=0, microsecond=0)],
                'ghi_avg': [current_avg],
                'ghi_min': [raw_df['ghi_wm2'].min()],
                'ghi_max': [raw_df['ghi_wm2'].max()],
                'sample_count': [len(raw_df)],
                'completeness': [len(raw_df)/12.0]
            })
            
            if df.empty:
                df = current_row
            else:
                # Add if not already there (avoid double counting if rollover just happened)
                if current_row['hour_timestamp'].iloc[0] not in df['hour_timestamp'].values:
                    df = pd.concat([df, current_row], ignore_index=True)
                    
        df = df.sort_values('hour_timestamp')
    except Exception as e:
        # st.error(f"Error loading sensor data: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

st.title("☀️ Solar Power Prediction & Management Dashboard")
st.markdown("Visualize actual vs predicted GHI (Global Horizontal Irradiance) for Bengaluru.")

with st.spinner("Loading Data..."):
    preds_df = load_predictions()
    sensor_df = load_sensor_data()

if preds_df.empty:
    st.warning("No prediction data available. Please run the inference pipeline first.")
    st.stop()

# Basic Setup
st.sidebar.header("Controls")
models_available = [m for m in preds_df.columns if m != 'timestamp']
selected_models = st.sidebar.multiselect("Select Models to Display", models_available, default=models_available[:1])

min_date = preds_df['timestamp'].min().date()
max_date = preds_df['timestamp'].max().date()
_today_date = datetime.now().date()

# Default to today if within range, else max_date
if min_date <= _today_date <= max_date:
    val = _today_date
else:
    val = max_date

selected_date = st.sidebar.date_input("Select Base Date for Forecast", min_value=min_date, max_value=max_date, value=val)

horizon = st.sidebar.radio("Select Forecast Horizon", ["24 Hours (Day 1)", "48 Hours (Day 1 & Day 2)", "72 Hours (Day 1, 2 & 3)"])

if "24" in horizon:
    hours = 24
elif "48" in horizon:
    hours = 48
else:
    hours = 72

start_dt = pd.to_datetime(selected_date)
end_dt = start_dt + timedelta(hours=hours)

mask = (preds_df['timestamp'] >= start_dt) & (preds_df['timestamp'] < end_dt)
filtered_preds = preds_df[mask]

# Define base colors for models
model_color_baselines = {
    'XGBoost': ['#636EFA', '#2E3BCC', '#00008B'],       # Blue hues (Day 1, 2, 3)
    'LightGBM': ['#EF553B', '#CC3218', '#8B0000'],      # Red hues 
    'RandomForest': ['#00CC96', '#009968', '#00663A'],  # Green hues 
    'ExtraTrees': ['#AB63FA', '#883BCC', '#4B0082'],    # Purple hues 
    'SVR': ['#FFA15A', '#CC752E', '#8B4500']            # Orange hues 
}

# ────────────── MAIN FORECAST CHART ──────────────
if len(selected_models) > 0:
    st.subheader(f"☀️ GHI Forecast starting {selected_date}")
    
    # Filter the data for the selected window
    mask = (preds_df['timestamp'] >= start_dt) & (preds_df['timestamp'] < end_dt)
    filtered_preds = preds_df[mask].sort_values('timestamp').drop_duplicates('timestamp')
    
    if filtered_preds.empty:
        st.warning(f"No prediction data found for {selected_date}. Earliest: {min_date}, Latest: {max_date}")
    else:
        fig = go.Figure()
        
        for model in selected_models:
            colors = model_color_baselines.get(model, ['#FFF'])
            
            # ONE continuous line for the whole horizon for maximum smoothness
            fig.add_trace(go.Scatter(
                x=filtered_preds['timestamp'], 
                y=filtered_preds[model],
                mode='lines',
                line_shape='spline',
                name=f'{model}',
                line=dict(color=colors[0], width=3),
                hovertemplate=f"<b>{model}</b><br>Time: %{{x}}<br>GHI: %{{y:.2f}} W/m²<extra></extra>"
            ))

        # Add Sensor Data if available in this range
        if not sensor_df.empty:
            s_mask = (sensor_df['hour_timestamp'] >= start_dt) & (sensor_df['hour_timestamp'] < end_dt)
            s_data = sensor_df[s_mask].sort_values('hour_timestamp').drop_duplicates('hour_timestamp')
            if not s_data.empty:
                fig.add_trace(go.Scatter(
                    x=s_data['hour_timestamp'],
                    y=s_data['ghi_avg'],
                    mode='lines+markers',
                    line_shape='spline',
                    name='Actual Sensor',
                    marker=dict(size=6, color='yellow'),
                    line=dict(color='yellow', width=3, dash='dot'),
                    hovertemplate="<b>Sensor Reading</b><br>Time: %{x}<br>GHI: %{y:.2f} W/m²<extra></extra>"
                ))
        
        # Add "NOW" line
        _now = datetime.now()
        if start_dt <= _now < end_dt:
            # Add the line without the buggy internal annotation logic
            fig.add_vline(x=_now, line_width=2, line_dash="dash", line_color="white")
            
            # Add the annotation separately to avoid the 'int + datetime/str' math error
            fig.add_annotation(
                x=_now, 
                y=1, 
                yref="paper",
                text="NOW", 
                showarrow=False, 
                xanchor="left", 
                yanchor="bottom",
                font=dict(color="white")
            )

        fig.update_layout(
            title="Hourly GHI Forecast (W/m²)",
            xaxis_title="Time", yaxis_title="GHI",
            hovermode="x unified", template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(rangeslider=dict(visible=True), type="date")
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select one or more models from the sidebar to view the forecast.")

st.markdown("---")
st.subheader("Model Validation & Sensor Comparison")
st.markdown("Comparison graph of sensor reading and predicted data for a 24-hour window.")

if not sensor_df.empty:
    c_start = pd.to_datetime(selected_date)
    c_end = c_start + timedelta(hours=24)
    
    # Use standard windowed data for comparison
    c_preds = preds_df[(preds_df['timestamp'] >= c_start) & (preds_df['timestamp'] < c_end)].sort_values('timestamp').drop_duplicates('timestamp')
    c_sensor = sensor_df[(sensor_df['hour_timestamp'] >= c_start) & (sensor_df['hour_timestamp'] < c_end)].sort_values('hour_timestamp').drop_duplicates('hour_timestamp')
    
    merged_comp = pd.merge(c_preds, c_sensor, left_on='timestamp', right_on='hour_timestamp', how='left')
    
    if not merged_comp.empty:
        merged_comp = merged_comp.sort_values('timestamp')
        
        comp_fig = go.Figure()
        comp_fig.add_trace(go.Scatter(
            x=merged_comp['timestamp'], y=merged_comp['ghi_avg'], 
            mode='lines+markers', line_shape='spline',
            name='Sensor (Actual)', line=dict(color='yellow', width=3)
        ))
        for model in selected_models:
            comp_fig.add_trace(go.Scatter(
                x=merged_comp['timestamp'], y=merged_comp[model], 
                mode='lines', line_shape='spline',
                name=f'{model} (Predicted)', 
                line=dict(color=model_color_baselines.get(model, ['#FFF'])[0], width=2)
            ))
            
        comp_fig.update_layout(
            title=f"24-Hour Comparison ({selected_date})", 
            xaxis_title="Time", yaxis_title="GHI (W/m²)", 
            template="plotly_dark", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(comp_fig, use_container_width=True)
        st.dataframe(merged_comp[['timestamp', 'ghi_avg'] + [m for m in selected_models if m in merged_comp.columns]].set_index('timestamp'))
    else:
        st.info("No data available for the selected comparison period.")
else:
    st.info("Run `sensor_reader.py` to gather live data for comparison.")
