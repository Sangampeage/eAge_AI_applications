import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import os

# -------------------------------
# CONFIG
# -------------------------------
# LATITUDE = 15.1394
# LONGITUDE = 76.9214
# LOCATION_NAME = "Ballari"
# LATITUDE = 16.1867
# LONGITUDE = 75.6961
# LOCATION_NAME = "Bagalkot"
# LATITUDE = 15.3350
# LONGITUDE = 75.0840
# LOCATION_NAME = "Hubali"
LATITUDE = float(input("Enter Latitude: "))
LONGITUDE = float(input("Enter Longitude: "))
LOCATION_NAME = input("Enter Location Name: ")  
FILE_PATH = "weather_soil_data.csv"

# -------------------------------
# API SETUP
# -------------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# -------------------------------
# API REQUEST
# -------------------------------
url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
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

# -------------------------------
# EXTRACT HOURLY DATA
# -------------------------------
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

# -------------------------------
# ADD METADATA
# -------------------------------
df["latitude"] = LATITUDE
df["longitude"] = LONGITUDE
df["location"] = LOCATION_NAME

# -------------------------------
# SAVE (APPEND + DEDUP)
# -------------------------------

# -------------------------------
# PREVIEW
# -------------------------------
print("Data collected successfully!")
print(df.head())
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
    else:        return "Normal"

df["risk"] = df.apply(classify_risk, axis=1)
print(df[["risk", "date", "temperature_2m", "humidity", "precipitation_probability", "soil_moisture"]].head())
if os.path.exists(FILE_PATH):
    existing = pd.read_csv(FILE_PATH)
    combined = pd.concat([existing, df], ignore_index=True)
    combined.drop_duplicates(subset=[ "date", "latitude", "longitude"], inplace=True)
    combined.to_csv(FILE_PATH, index=False)
else:
    df.to_csv(FILE_PATH, index=False)
df.to_csv(FILE_PATH, index=False)

