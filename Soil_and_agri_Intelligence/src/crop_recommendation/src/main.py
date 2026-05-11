# main.py

import os
import json
import logging
from inference import CropRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "rf_model.pkl")
    recommender = CropRecommender(model_path)

    input_json = {
        "soil": "Loamy",
        "N": 90,
        "P": 40,
        "K": 40,
        "ph": 6.5,
        "temperature": 28.0,
        "moisture": 70.0,
        "ec": 1.2
    }

    result_json = recommender.recommend(input_json)
    
    logger.info("Recommendation Output: %s", json.dumps(result_json, indent=2))