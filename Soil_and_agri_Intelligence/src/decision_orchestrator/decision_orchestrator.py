"""
decision_orchestrator.py
────────────────────────
Policy enforcement layer that sits between the ML models and the LLM advisory
engine.

Current active pipeline:
    crop_recommendation  ──►  DecisionOrchestrator  ──►  LLMAdvisoryEngine

Commented-out (future):
    disease_risk_model   ──►  DecisionOrchestrator  (enable when DB is ready)

The orchestrator:
  1. Receives crop recommendation output directly from CropRecommender.
  2. Applies threshold-based penalty policies (heat / flood / disease risk).
  3. Re-ranks crops by their adjusted scores.
  4. Emits a standardised advisory payload for the LLM engine.
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE RISK MODEL — commented out until DB integration is complete
# ─────────────────────────────────────────────────────────────────────────────
# from disease_risk_model import calculate_disease_risk
#
# Usage (when enabled):
#   disease_output = calculate_disease_risk(
#       crop_name=top_crop,
#       current_temperature=sensor_data["temperature"],
#       current_rainfall=sensor_data["rainfall"],
#       current_altitude=sensor_data["altitude"],
#   )
#   # disease_output keys: crop, risk_score, risk_level, stress_breakdown
# ─────────────────────────────────────────────────────────────────────────────


class DecisionOrchestrator:
    """
    Combines outputs from the crop recommendation model (and optionally the
    climate / disease risk models) to produce a final structured advisory.

    Args:
        db_connection: Optional psycopg2 connection for fetching crop tolerance
                       profiles.  Pass None when the DB is unavailable — the
                       orchestrator will still function using default tolerances.
    """

    # ── Policy thresholds ────────────────────────────────────────────────────
    FLOOD_RISK_TRIGGER   = 0.8   # activate flood penalty above this value
    DISEASE_RISK_TRIGGER = 0.7   # activate disease penalty above this value
    HEAT_RISK_TRIGGER    = 0.6   # activate heat penalty above this value

    FLOOD_PENALTY    = 0.80      # multiply crop score by this on flood risk
    DISEASE_PENALTY  = 0.85      # multiply ALL crop scores on disease risk
    HEAT_PENALTY     = 0.90      # multiply crop score by this on heat risk

    def __init__(self, db_connection=None):
        self.db = db_connection

    # ─────────────────────────────────────────────────────────────────────────
    # DB HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_crop_profiles(self, crop_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetches heat_tolerance and flood_tolerance for the given crops.
        Returns an empty dict if no DB connection is available.
        """
        if not self.db or not crop_names:
            return {}

        profiles: Dict[str, Dict[str, float]] = {}
        try:
            placeholders = ", ".join(["%s"] * len(crop_names))
            query = f"""
                SELECT crop_name, heat_tolerance, flood_tolerance
                FROM crop_profiles
                WHERE lower(crop_name) IN ({placeholders});
            """
            params = tuple(name.lower() for name in crop_names)

            with self.db.cursor() as cursor:
                cursor.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row)) if isinstance(row, (tuple, list)) else dict(row)
                    profiles[row_dict["crop_name"].lower()] = {
                        "heat_tolerance":  float(row_dict.get("heat_tolerance",  1.0)),
                        "flood_tolerance": float(row_dict.get("flood_tolerance", 1.0)),
                    }
        except Exception as exc:
            logger.error("Error fetching crop profiles: %s", exc)

        return profiles

    # ─────────────────────────────────────────────────────────────────────────
    # SCORE ADJUSTMENT
    # ─────────────────────────────────────────────────────────────────────────

    def adjust_crop_scores(
        self,
        crop_list: List[Dict[str, Any]],
        climate_risk: Dict[str, Any],
        disease_risk: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Applies threshold-based penalty rules to adjust crop confidence scores.

        Args:
            crop_list:    List of {"crop": str, "score": float} dicts from the
                          recommendation model.
            climate_risk: Dict with keys heat_risk, drought_risk, flood_risk
                          (all floats 0-1).  Pass {} if not available.
            disease_risk: Dict with key risk_score (float 0-1).
                          Pass {} if disease_risk_model is disabled.

        Returns:
            (adjusted_crops, alerts) tuple.
        """
        adjusted = copy.deepcopy(crop_list)
        alerts: List[str] = []

        crop_names = [c["crop"] for c in adjusted]
        profiles = self._fetch_crop_profiles(crop_names)

        heat_risk    = float(climate_risk.get("heat_risk",    0.0))
        flood_risk   = float(climate_risk.get("flood_risk",   0.0))
        disease_score = float(disease_risk.get("risk_score",  0.0))

        # ── Policy 1: Flood risk ──────────────────────────────────────────────
        if flood_risk >= self.FLOOD_RISK_TRIGGER:
            alerts.append(f"High flood risk detected (flood_risk={flood_risk:.2f})")
            for crop in adjusted:
                flood_tol = profiles.get(crop["crop"].lower(), {}).get("flood_tolerance", 1.0)
                if flood_risk > flood_tol:
                    crop["score"] *= self.FLOOD_PENALTY
                    logger.debug("Flood penalty applied to %s", crop["crop"])

        # ── Policy 2: Disease risk ────────────────────────────────────────────
        # NOTE: This policy will become more granular once disease_risk_model
        #       is re-enabled and provides per-crop disease scores.
        if disease_score >= self.DISEASE_RISK_TRIGGER:
            alerts.append(f"High disease susceptibility detected (risk_score={disease_score:.2f})")
            for crop in adjusted:
                crop["score"] *= self.DISEASE_PENALTY

        # ── Policy 3: Heat risk ───────────────────────────────────────────────
        if heat_risk > self.HEAT_RISK_TRIGGER:
            alerts.append(f"Elevated heat stress risk (heat_risk={heat_risk:.2f})")
            for crop in adjusted:
                heat_tol = profiles.get(crop["crop"].lower(), {}).get("heat_tolerance", 1.0)
                if heat_risk > heat_tol:
                    crop["score"] *= self.HEAT_PENALTY
                    logger.debug("Heat penalty applied to %s", crop["crop"])

        return adjusted, alerts

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIDENCE
    # ─────────────────────────────────────────────────────────────────────────

    def compute_decision_confidence(self, risks: List[float]) -> float:
        """Returns 1 - mean(risks), clamped to [0, 1]."""
        if not risks:
            return 1.0
        avg = sum(risks) / len(risks)
        return round(max(0.0, min(1.0, 1.0 - avg)), 2)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────────

    def orchestrate(
        self,
        crop_recommendation_output: Dict[str, Any],
        climate_risk: Optional[Dict[str, Any]] = None,
        disease_risk: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrates the full decision pipeline.

        Args:
            crop_recommendation_output: Direct output from CropRecommender.recommend().
                Must contain "top_crops": [{"crop": str, "score": float}, ...].
            climate_risk:  Optional climate risk dict
                           {"heat_risk": float, "drought_risk": float, "flood_risk": float}.
                           Defaults to all-zero (no risk) if omitted.
            disease_risk:  Optional disease risk dict {"risk_score": float}.
                           Currently disabled — pass None or omit.
                           ─────────────────────────────────────────────────
                           NOTE: disease_risk_model integration is COMMENTED
                           OUT.  To re-enable:
                             1. Uncomment the import at the top of this file.
                             2. Call calculate_disease_risk() here.
                             3. Pass the result as disease_risk.
                           ─────────────────────────────────────────────────

        Returns:
            Structured advisory payload ready for LLMAdvisoryEngine.
        """
        climate_risk  = climate_risk  or {}
        disease_risk  = disease_risk  or {}

        crop_list = crop_recommendation_output.get("top_crops", [])
        if not crop_list:
            logger.warning("crop_recommendation_output contains no top_crops.")

        # Apply penalty policies
        adjusted_crops, alerts = self.adjust_crop_scores(crop_list, climate_risk, disease_risk)

        # Re-rank by adjusted score, keep top 5
        adjusted_crops.sort(key=lambda x: x["score"], reverse=True)
        final_crops = [
            {"crop": c["crop"], "final_score": round(c["score"], 4)}
            for c in adjusted_crops[:5]
        ]

        # Aggregate risk values for confidence calculation
        heat_risk    = float(climate_risk.get("heat_risk",    0.0))
        drought_risk = float(climate_risk.get("drought_risk", 0.0))
        flood_risk   = float(climate_risk.get("flood_risk",   0.0))
        disease_score = float(disease_risk.get("risk_score",  0.0))

        confidence = self.compute_decision_confidence(
            [heat_risk, drought_risk, flood_risk, disease_score]
        )

        return {
            "recommended_crops": final_crops,
            "risk_summary": {
                "heat_risk":    heat_risk,
                "drought_risk": drought_risk,
                "flood_risk":   flood_risk,
                "disease_risk": disease_score,
            },
            "alerts":              alerts,
            "decision_confidence": confidence,
        }