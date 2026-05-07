"""
crop_recommendation
───────────────────
Top-level package. Re-exports the public API from the src sub-package so
callers can use either:

    from crop_recommendation import CropRecommender
    from crop_recommendation.src import CropRecommender   # also works
"""

from .src import CropRecommender, validate_input, SoilInput, CropPrediction

__all__ = [
    "CropRecommender",
    "validate_input",
    "SoilInput",
    "CropPrediction",
]