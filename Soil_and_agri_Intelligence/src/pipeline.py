"""
pipeline.py
───────────
Master integration script for the Soil & Agri Intelligence platform.

Active pipeline flow:
┌─────────────────────┐
│  Sensor / API Input │  (raw JSON from IoT sensors or API call)
└────────┬────────────┘
         │  validate + infer
         ▼
┌─────────────────────────┐
│  CropRecommender        │  (Random Forest / XGBoost ML model)
│  crop_recommendation    │  → top_crops with confidence scores
└────────┬────────────────┘
         │  structured dict
         ▼
┌─────────────────────────┐
│  DecisionOrchestrator   │  (penalty policies, risk adjustments, re-ranking)
│  decision_orchestrator  │  → advisory payload
└────────┬────────────────┘
         │  advisory payload
         ▼
┌─────────────────────────┐
│  LLMAdvisoryEngine      │  (Groq / Llama 3.3-70B via LangChain)
│  llm_reasoner           │  → 5-section farmer-friendly report
└─────────────────────────┘

Disease Risk Model (disease_risk_model):
  ─ Currently DISABLED / commented out.
  ─ Will plug in between CropRecommender and DecisionOrchestrator once the
    PostgreSQL DB is seeded with crop_thresholds data.
  ─ See DecisionOrchestrator.orchestrate() docstring for re-enable steps.
"""

import logging
import sys
import os
from typing import Any, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# Add src/ to sys.path so all sibling packages (crop_recommendation,
# decision_orchestrator, llm_reasoner, disease_risk_model) resolve correctly
# when this script is run directly:  python pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from crop_recommendation.src import CropRecommender       # noqa: E402
from decision_orchestrator import DecisionOrchestrator    # noqa: E402
from llm_reasoner import LLMAdvisoryEngine                # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE RISK MODEL — disabled until DB is ready
# ─────────────────────────────────────────────────────────────────────────────
# from disease_risk_model import calculate_disease_risk
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


class AgriculturalPipeline:
    """
    Binds CropRecommender → DecisionOrchestrator → LLMAdvisoryEngine into a
    single callable pipeline.

    Args:
        model_path:    Path to the trained .pkl model file.
                       Defaults to crop_recommendation/src/artifacts/rf_model.pkl.
        db_connection: Optional psycopg2 connection for crop tolerance lookups
                       in the orchestrator.  Pass None to skip DB-backed
                       penalty refinement.
    """

    def __init__(self, model_path: str = None, db_connection=None):
        self.recommender   = CropRecommender(model_path=model_path)
        self.orchestrator  = DecisionOrchestrator(db_connection=db_connection)
        self.llm_engine    = LLMAdvisoryEngine()
        logger.info("AgriculturalPipeline initialised.")

    def run(
        self,
        sensor_input: Dict[str, Any],
        climate_risk: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full pipeline end-to-end.

        Args:
            sensor_input:  Raw sensor payload dict.  Example:
                           {
                               "soil": "Loamy",
                               "N": 90, "P": 40, "K": 40,
                               "ph": 6.5, "temperature": 28.0,
                               "moisture": 70.0, "ec": 1.2
                           }
            climate_risk:  Optional external climate risk scores.  Example:
                           {"heat_risk": 0.3, "drought_risk": 0.2, "flood_risk": 0.1}
                           Defaults to all-zero (no active climate risk).

        Returns:
            A structured dict with status and data:
            {
                "status": "success",
                "message": "...",
                "data": {
                    "crop_recommendation": <raw model output>,
                    "orchestrator_output": <adjusted advisory payload>,
                    "advisory_report":     <LLM-generated text report>
                }
            }
        """
        try:
            climate_risk = climate_risk or {}

        # ── Step 1: Crop Recommendation ───────────────────────────────────────
            logger.info("Step 1 — Running CropRecommender …")
            crop_output_resp = self.recommender.recommend(sensor_input)
            if crop_output_resp.get("status") != "success":
                return crop_output_resp
                
            crop_output = crop_output_resp["data"]
            logger.info(
                "Recommendation: top crop = %s (score=%.4f)",
                crop_output["top_crops"][0]["crop"] if crop_output.get("top_crops") else "N/A",
                crop_output["top_crops"][0]["score"] if crop_output.get("top_crops") else 0.0,
            )

            # ─────────────────────────────────────────────────────────────────────
            # DISEASE RISK MODEL (disabled — re-enable block below when DB ready)
            # ─────────────────────────────────────────────────────────────────────
            # top_crop_name = crop_output["top_crops"][0]["crop"]
            # disease_output = calculate_disease_risk(
            #     crop_name=top_crop_name,
            #     current_temperature=sensor_input["temperature"],
            #     current_rainfall=sensor_input.get("rainfall", 0),
            #     current_altitude=sensor_input.get("altitude", 0),
            # )
            # disease_risk = {"risk_score": disease_output["risk_score"]}
            # logger.info(
            #     "Disease risk for %s: %.2f (%s)",
            #     top_crop_name, disease_output["risk_score"], disease_output["risk_level"]
            # )
            disease_risk: Dict[str, Any] = {}   # placeholder until DB enabled
            # ─────────────────────────────────────────────────────────────────────

            # ── Step 2: Decision Orchestration ────────────────────────────────────
            logger.info("Step 2 — Running DecisionOrchestrator …")
            orchestrator_output = self.orchestrator.orchestrate(
                crop_recommendation_output=crop_output,
                climate_risk=climate_risk,
                disease_risk=disease_risk,
            )
            logger.info(
                "Orchestration complete. Confidence=%.2f, Alerts=%s",
                orchestrator_output["decision_confidence"],
                orchestrator_output["alerts"],
            )

            # ── Step 3: LLM Advisory ──────────────────────────────────────────────
            logger.info("Step 3 — Generating LLM advisory …")
            advisory_text = self.llm_engine.generate_advisory(orchestrator_output)

            return {
                "status": "success",
                "message": "Pipeline completed successfully",
                "data": {
                    "crop_recommendation": crop_output,
                    "orchestrator_output": orchestrator_output,
                    "advisory_report":     advisory_text,
                }
            }
            
        except Exception as exc:
            logger.exception("Pipeline error: %s", exc)
        return {
            "status": "failed",
            "message": str(exc),
            "data": None
        }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — quick smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    # Example sensor payload
    sample_sensor_input = {
        "soil":        "Loamy",
        "N":           90.0,
        "P":           40.0,
        "K":           40.0,
        "ph":           6.5,
        "temperature": 28.0,
        "moisture":    70.0,
        "ec":           1.2,
        "location":    "Samastipur"
    }

    # Optional climate risk from external weather API / climate model
    sample_climate_risk = {
        "heat_risk":    0.3,
        "drought_risk": 0.2,
        "flood_risk":   0.1,
    }

    pipeline = AgriculturalPipeline()   # uses default artifact paths, no DB

    try:
        result = pipeline.run(
            sensor_input=sample_sensor_input,
            climate_risk=sample_climate_risk,
        )

        logger.info("Pipeline Execution Completed")
        logger.info("Final Response: %s", json.dumps(result, indent=2))

    except Exception as exc:
        logger.exception("Pipeline execution failed: %s", exc)