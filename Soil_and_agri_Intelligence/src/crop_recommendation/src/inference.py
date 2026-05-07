"""
inference.py
────────────
CropRecommender: the single public entry point for the crop recommendation
module. Accepts raw sensor JSON, validates it, runs the ML model, and returns
a structured dict ready for the Decision Orchestrator.

Output contract (pipeline-compatible):
{
    "model": "crop_recommendation",
    "top_crops": [
        {"crop": "Maize",  "score": 0.87},
        {"crop": "Sorghum","score": 0.73},
        {"crop": "Rice",   "score": 0.61}
    ],
    "raw_recommended_crops": [           # same data, confidence key preserved
        {"crop": "Maize",  "confidence": 0.87},
        ...
    ]
}
"""

import json
import logging
import os
from typing import Dict, Any, Union

import joblib
import numpy as np

from schemas import SoilInput
from validation import validate_input
from preprocessing import load_encoders
from model_wrapper import CropModel

logger = logging.getLogger(__name__)

_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


class CropRecommender:
    """
    End-to-end crop recommendation: sensor input → validated → ML → structured output.
    """

    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Path to the trained model .pkl file.
                        Defaults to artifacts/rf_model.pkl relative to this file.
        """
        if model_path is None:
            model_path = os.path.join(_ARTIFACTS_DIR, "rf_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at: {model_path}. "
                "Run train.py first to generate the model."
            )

        self.model = joblib.load(model_path)
        self.soil_encoder, self.label_encoder = load_encoders()
        self.model_wrapper = CropModel(self.model, self.label_encoder)
        logger.info("CropRecommender initialised with model: %s", model_path)

    # ── feature engineering ──────────────────────────────────────────────────

    def _build_features(self, data: SoilInput) -> np.ndarray:
        """Encodes soil type and concatenates with numeric sensor readings."""
        soil_encoded = self.soil_encoder.transform([[data.soil]])[0]
        numeric = np.array([
            data.N,
            data.P,
            data.K,
            data.ph,
            data.temperature,
            data.moisture,
        ])
        return np.concatenate([soil_encoded, numeric])

    # ── public API ────────────────────────────────────────────────────────────

    def recommend(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Full inference pipeline: validate → preprocess → predict → format.

        Args:
            input_data: Raw sensor payload as a dict or JSON string.

        Returns:
            Pipeline-compatible dict with key "model" == "crop_recommendation".

        Raises:
            ValueError:        On validation / threshold failures (bad sensor data).
            FileNotFoundError: If model artifacts are missing.
        """
        # ── parse JSON string if needed ──────────────────────────────────────
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Input is not valid JSON: {exc}. "
                    "Provide a well-formed JSON string or a Python dict."
                ) from exc

        # ── validate & threshold-check ───────────────────────────────────────
        validated: SoilInput = validate_input(input_data)

        # ── feature engineering ───────────────────────────────────────────────
        features = self._build_features(validated)

        # ── model inference ───────────────────────────────────────────────────
        predictions = self.model_wrapper.predict(features)

        # ── build pipeline-compatible output ─────────────────────────────────
        top_crops_for_orchestrator = [
            {"crop": p.crop, "score": round(p.confidence, 4)}
            for p in predictions
        ]
        raw_crops = [
            {"crop": p.crop, "confidence": round(p.confidence, 4)}
            for p in predictions
        ]

        result = {
            "model": "crop_recommendation",
            "top_crops": top_crops_for_orchestrator,   # consumed by orchestrator
            "raw_recommended_crops": raw_crops,         # full detail for logging/debug
        }

        logger.info(
            "Recommendation complete. Top crop: %s (%.4f)",
            predictions[0].crop,
            predictions[0].confidence,
        )
        return result

    def recommend_json(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """Same as recommend() but returns a pretty-printed JSON string."""
        return json.dumps(self.recommend(input_data), indent=2)