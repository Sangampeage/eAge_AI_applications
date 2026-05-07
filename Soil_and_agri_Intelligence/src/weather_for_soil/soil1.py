import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta

# -------------------------------
# API SETUP
# -------------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def get_average_risk(latitude, longitude):

    # Last 7 days
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=7)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "soil_moisture_0_to_1cm"
        ],
        "timezone": "Asia/Kolkata"
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()

    # -------------------------------
    # CREATE DATAFRAME
    # -------------------------------
    df = pd.DataFrame({
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "humidity": hourly.Variables(1).ValuesAsNumpy(),
        "precipitation_probability": hourly.Variables(2).ValuesAsNumpy(),
        "soil_moisture": hourly.Variables(3).ValuesAsNumpy(),
    })

    # -------------------------------
    # RISK CLASSIFICATION
    # -------------------------------
    def classify_risk(row):

        temp = row["temperature_2m"]
        humidity = row["humidity"]
        rain_prob = row["precipitation_probability"]
        soil_moisture = row["soil_moisture"]

        # Flood Risk
        if soil_moisture > 0.35 and rain_prob > 70 and humidity > 80:
            return "Flood Risk"

        # Drought Risk
        elif soil_moisture < 0.15 and rain_prob < 20 and temp > 30:
            return "Drought Risk"

        # Heat Stress
        elif temp > 35 and humidity < 60:
            return "Heat Stress"

        else:
            return "Normal"

    df["risk"] = df.apply(classify_risk, axis=1)

    # -------------------------------
    # MOST COMMON RISK
    # -------------------------------
    most_common_risk = df["risk"].mode()[0]

    return most_common_risk


# -------------------------------
# EXAMPLE USAGE
# -------------------------------
risk = get_average_risk(15.3350, 75.0840)

print("Average Risk:", risk)