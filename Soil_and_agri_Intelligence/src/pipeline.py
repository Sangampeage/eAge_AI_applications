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
# Ensure all sub-packages resolve correctly regardless of how the script is
# invoked (python pipeline.py  vs  python -m src.pipeline).
# ─────────────────────────────────────────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# crop_recommendation uses its own internal imports; add src/ to path so they
# resolve when imported as a package from pipeline.py.
_CROP_SRC = os.path.join(_SRC_DIR, "crop_recommendation", "src")
if _CROP_SRC not in sys.path:
    sys.path.insert(0, _CROP_SRC)

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
            {
                "crop_recommendation": <raw model output>,
                "orchestrator_output": <adjusted advisory payload>,
                "advisory_report":     <LLM-generated text report>
            }

        Raises:
            ValueError:        On sensor validation / threshold failure.
            FileNotFoundError: If ML model artifacts are missing.
        """
        climate_risk = climate_risk or {}

        # ── Step 1: Crop Recommendation ───────────────────────────────────────
        logger.info("Step 1 — Running CropRecommender …")
        crop_output = self.recommender.recommend(sensor_input)
        logger.info(
            "Recommendation: top crop = %s (score=%.4f)",
            crop_output["top_crops"][0]["crop"] if crop_output["top_crops"] else "N/A",
            crop_output["top_crops"][0]["score"] if crop_output["top_crops"] else 0.0,
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
            "crop_recommendation": crop_output,
            "orchestrator_output": orchestrator_output,
            "advisory_report":     advisory_text,
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

        print("\n" + "═" * 70)
        print("CROP RECOMMENDATION OUTPUT")
        print("═" * 70)
        print(json.dumps(result["crop_recommendation"], indent=2))

        print("\n" + "═" * 70)
        print("ORCHESTRATOR OUTPUT")
        print("═" * 70)
        print(json.dumps(result["orchestrator_output"], indent=2))

        print("\n" + "═" * 70)
        print("LLM ADVISORY REPORT")
        print("═" * 70)
        print(result["advisory_report"])

    except ValueError as ve:
        logger.error("Sensor validation failed: %s", ve)
    except FileNotFoundError as fe:
        logger.error("Model artifact missing: %s", fe)
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)