"""
validation.py
─────────────
Validates raw sensor input before it reaches the ML model.

Two layers of defense:
  1. Structural validation  — are all fields present and the right type?
  2. Domain / threshold validation — are numeric values within agronomically
     plausible ranges?

Thresholds are deliberately wide (field conditions vary greatly worldwide) but
will reject clear sensor faults such as negative NPK readings, sub-zero
absolute moisture, or temperatures that would vaporise water.

Raises ValueError with a descriptive, human-readable message on any failure.
"""

import logging
from typing import Any, Dict

from schemas import SoilInput

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# REQUIRED FIELDS
# ──────────────────────────────────────────────────────────────────────────────
REQUIRED_FIELDS = ["soil", "N", "P", "K", "ph", "temperature", "moisture", "ec"]

# ──────────────────────────────────────────────────────────────────────────────
# SENSOR THRESHOLDS
# Each entry: (min_inclusive, max_inclusive, unit_hint)
# Set to None to skip that bound.
# ──────────────────────────────────────────────────────────────────────────────
SENSOR_THRESHOLDS: Dict[str, tuple] = {
    # Macronutrients (mg/kg or ppm — typical soil test range)
    "N":           (0.0,   500.0,  "mg/kg"),
    "P":           (0.0,   300.0,  "mg/kg"),
    "K":           (0.0,   1000.0, "mg/kg"),

    # Soil pH (standard 0–14 scale; practical crop range 3.5–9.5)
    "ph":          (3.5,   9.5,    "pH units"),

    # Air / soil surface temperature (°C)
    "temperature": (-10.0, 60.0,   "°C"),

    # Volumetric water content or relative humidity (%)
    "moisture":    (0.0,   100.0,  "%"),

    # Electrical conductivity (dS/m — above ~8 most crops fail)
    "ec":          (0.0,   8.0,    "dS/m"),
}

# Valid soil types (lowercase for matching; extend as needed)
VALID_SOIL_TYPES = {
    "loamy", "sandy", "clay", "silty", "peaty",
    "chalky", "saline", "black", "red", "laterite",
    "alluvial", "sandy loam", "clay loam", "silt loam",
}


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _check_missing_fields(data: dict) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(
            f"Missing required sensor field(s): {missing}. "
            f"Expected fields are: {REQUIRED_FIELDS}"
        )


def _check_soil_type(soil_value: Any) -> str:
    if not isinstance(soil_value, str):
        raise ValueError(
            f"Field 'soil' must be a string (e.g. 'Loamy'), "
            f"got {type(soil_value).__name__}: {soil_value!r}"
        )
    cleaned = soil_value.strip()
    if not cleaned:
        raise ValueError("Field 'soil' must not be an empty string.")
    if cleaned.lower() not in VALID_SOIL_TYPES:
        logger.warning(
            "Soil type '%s' is not in the known list %s. "
            "Proceeding, but verify sensor data.",
            cleaned, sorted(VALID_SOIL_TYPES)
        )
    return cleaned


def _check_numeric_fields(data: dict) -> Dict[str, float]:
    """
    Validates that every numeric field is:
      - An int or float (not a string, None, NaN, etc.)
      - Within the configured sensor threshold range
    Returns a dict of field -> float-cast value.
    """
    numeric_fields = REQUIRED_FIELDS[1:]  # everything except 'soil'
    result: Dict[str, float] = {}

    for field in numeric_fields:
        raw = data[field]

        # ── type check ──────────────────────────────────────────────────────
        if not isinstance(raw, (int, float)):
            raise ValueError(
                f"Field '{field}' must be a numeric value (int or float), "
                f"got {type(raw).__name__}: {raw!r}. "
                "Check sensor output for string encoding or null values."
            )

        value = float(raw)

        # ── NaN / Inf check ──────────────────────────────────────────────────
        import math
        if math.isnan(value):
            raise ValueError(
                f"Field '{field}' contains NaN — sensor may be offline or "
                "returning an error code."
            )
        if math.isinf(value):
            raise ValueError(
                f"Field '{field}' contains an infinite value — "
                "likely a sensor hardware fault."
            )

        # ── range / threshold check ──────────────────────────────────────────
        if field in SENSOR_THRESHOLDS:
            lo, hi, unit = SENSOR_THRESHOLDS[field]
            if lo is not None and value < lo:
                raise ValueError(
                    f"Field '{field}' = {value} {unit} is BELOW the minimum "
                    f"acceptable threshold of {lo} {unit}. "
                    "Possible sensor fault or data corruption."
                )
            if hi is not None and value > hi:
                raise ValueError(
                    f"Field '{field}' = {value} {unit} is ABOVE the maximum "
                    f"acceptable threshold of {hi} {unit}. "
                    "Possible sensor fault or extreme outlier — verify reading."
                )

        result[field] = value

    return result


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def validate_input(data: dict) -> SoilInput:
    """
    Validates a raw sensor/API input dictionary and returns a typed SoilInput.

    Validation order:
      1. All required fields are present.
      2. 'soil' is a non-empty string (warns if unknown type).
      3. All numeric fields are valid numbers within sensor thresholds.

    Args:
        data: Raw input dictionary (e.g. parsed from JSON / sensor payload).

    Returns:
        SoilInput dataclass with validated, type-cast values.

    Raises:
        ValueError: With a descriptive message identifying the exact problem.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"Input must be a dictionary, got {type(data).__name__}. "
            "Ensure the sensor payload is parsed as JSON before validation."
        )

    # Step 1 — structural completeness
    _check_missing_fields(data)

    # Step 2 — soil type
    soil = _check_soil_type(data["soil"])

    # Step 3 — numeric ranges
    numeric = _check_numeric_fields(data)

    logger.debug("Sensor input validated successfully for soil type '%s'.", soil)

    return SoilInput(
        soil=soil,
        N=numeric["N"],
        P=numeric["P"],
        K=numeric["K"],
        ph=numeric["ph"],
        temperature=numeric["temperature"],
        moisture=numeric["moisture"],
        ec=numeric["ec"],
    )