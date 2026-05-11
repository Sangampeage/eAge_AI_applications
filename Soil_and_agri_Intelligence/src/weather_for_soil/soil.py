import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os

import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# API SETUP
# -------------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def classify_risk(row):    
    temp = row["temperature_2m"]    
    humidity = row["humidity"]   
    rain_prob = row["precipitation_probability"]    
    soil_moisture = row["soil_moisture"]    
    # Flood    
    if soil_moisture > 0.35 and rain_prob > 70 and humidity > 80:        
        return "Flood Risk"    
    # Drought    
    elif soil_moisture < 0.15 and rain_prob < 20 and temp > 30:       
        return "Drought Risk"   
    # Heat Stress   
    elif temp > 35 and humidity < 60:       
        return "Heat Stress"    
    else:        
        return "Normal"

def fetch_weather_soil_data(latitude: float, longitude: float, location_name: str, file_path: str = "weather_soil_data.csv"):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation_probability",
                "soil_temperature_0cm",
                "soil_moisture_0_to_1cm"
            ],
            "timezone": "Asia/Kolkata"
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        hourly = response.Hourly()

        data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "precipitation_probability": hourly.Variables(2).ValuesAsNumpy(),
            "soil_temperature": hourly.Variables(3).ValuesAsNumpy(),
            "soil_moisture": hourly.Variables(4).ValuesAsNumpy(),
        }

        df = pd.DataFrame(data)
        df["latitude"] = latitude
        df["longitude"] = longitude
        df["location"] = location_name
        df["risk"] = df.apply(classify_risk, axis=1)

        if os.path.exists(file_path):
            existing = pd.read_csv(file_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.drop_duplicates(subset=[ "date", "latitude", "longitude"], inplace=True)
            combined.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)
            
        return {
            "status": "success",
            "message": "Data collected and saved successfully.",
            "data": {
                "file_path": file_path,
                "records_processed": len(df)
            }
        }
    except Exception as e:
        logger.error("Error fetching weather soil data: %s", str(e))
        return {
            "status": "failed",
            "message": str(e),
            "data": None
        }

if __name__ == "__main__":
    # Example Usage
    try:
        lat = 15.3350
        lon = 75.0840
        loc = "Hubali"
        result = fetch_weather_soil_data(lat, lon, loc)
        logger.info("Result: %s", json.dumps(result, indent=2))
    except Exception as e:
        logger.error("Failed to run fetch_weather_soil_data: %s", str(e))
