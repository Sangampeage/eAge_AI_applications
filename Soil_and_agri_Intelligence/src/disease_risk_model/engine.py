from typing import Dict, Any
from .config_loader import CropConfigLoader
import logging

logger = logging.getLogger(__name__)

def get_risk_level(score: float) -> str:
    """
    Map risk score to a categorical risk level based on defined thresholds.
    """
    if score <= 0.3:
        return "Low"
    elif score <= 0.6:
        return "Moderate"
    elif score <= 0.8:
        return "High"
    else:
        return "Critical"

def calculate_disease_risk(
    crop_name: str, 
    current_temperature: float, 
    current_rainfall: float, 
    current_altitude: float
) -> Dict[str, Any]:
    """
    Calculates the disease/abnormality risk for a crop based on environmental deviations
    from its optimal/expected configurations stored in the database.
    
    Args:
        crop_name (str): Name of the crop to analyze.
        current_temperature (float): Current observed temperature.
        current_rainfall (float): Current observed rainfall.
        current_altitude (float): Current altitude.
        
    Returns:
        Dict[str, Any]: A JSON-serializable dictionary containing risk metrics.
    """
    # 1. Load configuration dynamically from the DB
    config = CropConfigLoader.get_crop_thresholds(crop_name)
    
    if not config:
        raise ValueError(f"Configuration for crop '{crop_name}' not found in the database.")
        
    # Extract thresholds dynamically
    t_min = config['temp_min']
    t_max = config['temp_max']
    t_opt = config['temp_optimum']
    r_expected = config['rainfall_expected']
    a_expected = config['altitude_expected']
    
    # 2. Temperature deviation calculation
    if t_min <= current_temperature <= t_max:
        temp_range = t_max - t_min
        if temp_range == 0:
            temp_stress = 0.0
        else:
            temp_stress = abs(current_temperature - t_opt) / temp_range
    else:
        temp_stress = 1.0
        
    # Clamp temp stress between 0 and 1
    temp_stress = max(0.0, min(1.0, temp_stress))

    # 3. Rainfall deviation calculation
    if r_expected > 0:
        rainfall_stress = abs(current_rainfall - r_expected) / r_expected
    else:
        rainfall_stress = 1.0 if current_rainfall > 0 else 0.0
        
    # Clamp rainfall stress between 0 and 1
    rainfall_stress = max(0.0, min(1.0, rainfall_stress))
    
    # 4. Altitude mismatch calculation
    if a_expected > 0:
        altitude_stress = abs(current_altitude - a_expected) / a_expected
    else:
        altitude_stress = 1.0 if current_altitude > 0 else 0.0
        
    # Clamp altitude stress between 0 and 1
    altitude_stress = max(0.0, min(1.0, altitude_stress))
    
    # 5. Final Risk Score calculation
    risk_score = (0.5 * temp_stress) + (0.3 * rainfall_stress) + (0.2 * altitude_stress)
    
    # Final clamping just in case
    risk_score = max(0.0, min(1.0, risk_score))
    
    # 6. Determine risk level
    risk_level = get_risk_level(risk_score)
    
    # 7. Construct output JSON structure exactly as specified
    output = {
        "crop": config['crop_name'], # Use the official name from DB
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "stress_breakdown": {
            "temperature": round(temp_stress, 2),
            "rainfall": round(rainfall_stress, 2),
            "altitude": round(altitude_stress, 2)
        }
    }
    
    return output

if __name__ == "__main__":
    # Quick sanity check / example test
    try:
        # Note: requires the DB to be up and seeded
        result = calculate_disease_risk(
            crop_name="Maize",
            current_temperature=30.0,
            current_rainfall=500.0,
            current_altitude=1200.0
        )
        import json
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Test failed: {e}")
