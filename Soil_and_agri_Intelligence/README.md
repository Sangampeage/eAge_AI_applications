# Soil and Agri Intelligence

An intelligent agricultural decision-making platform integrating ML-based crop recommendation, rule-based risk orchestration, and LLM-generated farmer-friendly advisories.

---

## Active Pipeline

```
Sensor Input (JSON)
       │
       ▼
CropRecommender          ← crop_recommendation/src/
(RF / XGBoost model)
       │  top_crops with confidence scores
       ▼
DecisionOrchestrator     ← decision_orchestrator/
(penalty policies, re-ranking)
       │  advisory payload
       ▼
LLMAdvisoryEngine        ← llm_reasoner/
(Groq / Llama 3.3-70B)
       │
       ▼
5-Section Farmer Advisory Report
```

> **Disease Risk Model** (`disease_risk_model/`) is implemented and preserved but currently **disabled** pending PostgreSQL DB setup. See `pipeline.py` and `decision_orchestrator.py` for the commented-out re-enable blocks.

---

## Project Structure

```
Soil_and_agri_Intelligence/
├── .env.example                        ← copy to .env, fill credentials
├── README.md
├── data/
│   ├── crop_climate_requirements.csv   ← seeds disease_risk DB table
│   └── sensor_Crop_Dataset.csv         ← training data
└── src/
    ├── pipeline.py                     ← ENTRY POINT — runs the full pipeline
    ├── preprocess.py                   ← data exploration utility
    │
    ├── crop_recommendation/            ← ML recommendation module
    │   ├── __init__.py
    │   └── src/
    │       ├── __init__.py             ← exposes CropRecommender
    │       ├── schemas.py              ← SoilInput, CropPrediction dataclasses
    │       ├── validation.py           ← sensor threshold guards
    │       ├── preprocessing.py        ← encoders + feature engineering
    │       ├── model_wrapper.py        ← CropModel (top-N predictions)
    │       ├── inference.py            ← CropRecommender (public entry point)
    │       ├── train.py                ← model training script
    │       └── artifacts/              ← rf_model.pkl, encoders (git-ignored)
    │
    ├── disease_risk_model/             ← DB-backed risk engine (DISABLED)
    │   ├── __init__.py
    │   ├── schema.sql
    │   ├── db.py
    │   ├── config_loader.py
    │   ├── engine.py
    │   └── importer.py
    │
    ├── decision_orchestrator/          ← policy + re-ranking layer
    │   ├── __init__.py
    │   └── decision_orchestrator.py
    │
    └── llm_reasoner/                   ← Groq / LangChain advisory engine
        ├── __init__.py
        └── llm_advisory_engine.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install psycopg2-binary pandas scikit-learn xgboost joblib langchain-groq python-dotenv
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY at minimum; DB vars only needed when disease model is enabled
```

### 3. Train the model

```bash
cd src/crop_recommendation/src
python train.py --data ../../../data/Crop_recommendation_dataset.csv
```

### 4. Run the pipeline

```bash
cd src
python pipeline.py
```

---

## Sensor Input Schema

| Field         | Type   | Unit    | Valid Range          | Description              |
|---------------|--------|---------|----------------------|--------------------------|
| `soil`        | string | —       | See VALID_SOIL_TYPES | Soil classification      |
| `N`           | float  | mg/kg   | 0 – 500              | Nitrogen content         |
| `P`           | float  | mg/kg   | 0 – 300              | Phosphorus content       |
| `K`           | float  | mg/kg   | 0 – 1000             | Potassium content        |
| `ph`          | float  | pH      | 3.5 – 9.5            | Soil pH                  |
| `temperature` | float  | °C      | -10 – 60             | Air/soil temperature     |
| `moisture`    | float  | %       | 0 – 100              | Volumetric water content |
| `ec`          | float  | dS/m    | 0 – 8.0              | Electrical conductivity  |

Values outside these ranges will raise a `ValueError` with a descriptive message indicating the field, the received value, and the expected range.

---

## Re-enabling the Disease Risk Model

1. Set up PostgreSQL and run `src/disease_risk_model/schema.sql`.
2. Seed the DB: `python src/disease_risk_model/importer.py`.
3. In `src/pipeline.py`, uncomment the `calculate_disease_risk` import and the disease risk block inside `AgriculturalPipeline.run()`.
4. In `src/decision_orchestrator/decision_orchestrator.py`, uncomment the import at the top of the file.

---

## Output Structure

```json
{
  "crop_recommendation": {
    "model": "crop_recommendation",
    "top_crops": [{"crop": "Maize", "score": 0.87}, ...],
    "raw_recommended_crops": [{"crop": "Maize", "confidence": 0.87}, ...]
  },
  "orchestrator_output": {
    "recommended_crops": [{"crop": "Maize", "final_score": 0.87}],
    "risk_summary": {"heat_risk": 0.3, "drought_risk": 0.2, "flood_risk": 0.1, "disease_risk": 0.0},
    "alerts": [],
    "decision_confidence": 0.85
  },
  "advisory_report": "1. Recommended Crops Summary\n..."
}
```