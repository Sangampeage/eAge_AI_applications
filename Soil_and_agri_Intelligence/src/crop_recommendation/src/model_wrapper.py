"""
model_wrapper.py
────────────────
Thin wrapper around the trained sklearn / XGBoost model.
Exposes a clean predict() interface that returns the top-N crop predictions.
"""

import logging
from typing import List

import numpy as np

from .schemas import CropPrediction

logger = logging.getLogger(__name__)


class CropModel:
    """
    Wraps a trained probabilistic classifier and returns top-N predictions
    as typed CropPrediction objects.
    """

    def __init__(self, model, label_encoder, top_n: int = 3):
        """
        Args:
            model:          Fitted sklearn / XGBoost model with predict_proba().
            label_encoder:  Fitted LabelEncoder used during training.
            top_n:          Number of top crops to return (default 3).
        """
        self.model = model
        self.label_encoder = label_encoder
        self.top_n = top_n

    def predict(self, features: np.ndarray) -> List[CropPrediction]:
        """
        Args:
            features: 1-D numpy array of pre-processed features.

        Returns:
            List of CropPrediction sorted by confidence descending.
        """
        probs = self.model.predict_proba([features])[0]

        top_idx = np.argsort(probs)[-self.top_n:][::-1]
        crops = self.label_encoder.inverse_transform(top_idx)

        predictions = [
            CropPrediction(crop=str(crop), confidence=float(probs[i]))
            for crop, i in zip(crops, top_idx)
        ]

        logger.debug(
            "Top-%d predictions: %s",
            self.top_n,
            [(p.crop, round(p.confidence, 4)) for p in predictions],
        )
        return predictions