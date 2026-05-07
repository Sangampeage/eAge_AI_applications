"""
crop_recommendation.src
───────────────────────
Public API for the crop recommendation module.

Usage from pipeline:
    from crop_recommendation.src import CropRecommender

    recommender = CropRecommender()
    output = recommender.recommend(sensor_payload)  # returns pipeline-compatible dict
"""

from .inference import CropRecommender
from .validation import validate_input
from .schemas import SoilInput, CropPrediction

__all__ = [
    "CropRecommender",
    "validate_input",
    "SoilInput",
    "CropPrediction",
]