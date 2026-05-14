import requests
import logging

logger = logging.getLogger(__name__)

def get_complete_weather(city, country_code=None):
    """
    Fetches weather for a specific city. 
    Providing a 2-letter country_code (e.g., 'IN' for India) improves accuracy.
    """
    try:
        # 1. Geocoding: Get Latitude and Longitude
        geo_query = f"{city}"
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={geo_query}&count=5&language=en&format=json"
        
        geo_response = requests.get(geo_url).json()

        if "results" not in geo_response:
            return {"error": f"Could not find location '{city}'."}

        # Filter results by country if country_code is provided, else take the first result
        location = None
        if country_code:
            for res in geo_response["results"]:
                if res.get("country_code", "").upper() == country_code.upper():
                    location = res
                    break
        
        if not location:
            location = geo_response["results"][0]

        lat = location["latitude"]
        lon = location["longitude"]
        display_name = f"{location['name']}, {location.get('admin1', '')}, {location.get('country', '')}"

        # 2. Weather: Fetch current and daily forecast data
        weather_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "weather_code", "wind_speed_10m"],
            "daily": ["temperature_2m_max", "temperature_2m_min", "sunrise", "sunset"],
            "timezone": "auto" 
        }
        
        weather_data = requests.get(weather_url, params=params).json()

        # 3. Format and Return the Data
        current = weather_data["current"]
        daily = weather_data["daily"]

        return {
            "status": "success",
            "location": display_name,
            "latitude": lat,
            "longitude": lon,
            "current_temp": f"{current['temperature_2m']}°C",
            "feels_like": f"{current['apparent_temperature']}°C",
            "humidity": f"{current['relative_humidity_2m']}%",
            "wind_speed": f"{current['wind_speed_10m']} km/h",
            "today_high": f"{daily['temperature_2m_max'][0]}°C",
            "today_low": f"{daily['temperature_2m_min'][0]}°C",
            "sunrise": daily["sunrise"][0],
            "sunset": daily["sunset"][0]
        }
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # --- Usage ---
    city_input = "Samastipur"
    country_input = "IN" 
    
    report = get_complete_weather(city_input, country_input)
    
    if report.get("status") == "success":
        print(f"--- Weather Report for {report['location']} ---")
        print(f"Coordinates:  {report['latitude']}, {report['longitude']}")
        print(f"Current Temp: {report['current_temp']} (Feels like {report['feels_like']})")
        print(f"Humidity:     {report['humidity']}")
        print(f"Wind Speed:   {report['wind_speed']}")
        print(f"High/Low:     {report['today_high']} / {report['today_low']}")
    else:
        print(f"Error: {report.get('error')}")
