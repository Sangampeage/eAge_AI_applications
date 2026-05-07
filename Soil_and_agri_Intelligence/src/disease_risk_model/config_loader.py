from .db import execute_query
import logging

logger = logging.getLogger(__name__)

class CropConfigLoader:
    """
    Handles loading of crop configurations from the database.
    This separates the data fetching logic from the risk calculation logic.
    """
    
    @staticmethod
    def get_crop_thresholds(crop_name: str) -> dict:
        """
        Fetches the threshold values for a specific crop from the database.
        
        Args:
            crop_name (str): The name of the crop (case-insensitive match).
            
        Returns:
            dict: The configuration dictionary for the crop, or None if not found.
        """
        query = """
            SELECT 
                crop_name, temp_min, temp_max, temp_optimum, 
                rainfall_expected, altitude_expected
            FROM crop_thresholds 
            WHERE lower(crop_name) = lower(%s);
        """
        
        results = execute_query(query, (crop_name,), fetch=True)
        if results:
            return dict(results[0])
        
        logger.warning(f"No configuration found in database for crop: {crop_name}")
        return None
