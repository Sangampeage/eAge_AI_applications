import os
import json
import logging
import csv
from typing import Dict, Any, List

from dotenv import load_dotenv

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    ChatGroq = None
    SystemMessage = None
    HumanMessage = None

logger = logging.getLogger(__name__)

class LLMAdvisoryEngine:
    """
    LLM Integration Module that generates farmer-friendly advisory text based on
    structured outputs from the Decision Orchestrator using Groq.
    
    Now incorporates regional suitability data from crop_soil_suitability.csv.
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        load_dotenv()
        self.model_name = model_name
        
        if ChatGroq is None:
            logger.warning("Langchain Groq package not found. Please install with: pip install langchain-groq python-dotenv")
            self.model = None
        else:
            self.model = ChatGroq(model=self.model_name, temperature=0.2)
            
        # Load regional suitability data
        self.suitability_data = self._load_suitability_data()

    def _load_suitability_data(self) -> List[Dict[str, str]]:
        """Loads the crop suitability data from the data folder."""
        from pathlib import Path
        
        data = []
        
        # Container-friendly: use env var or resolve relative to this file
        data_dir_env = os.environ.get("DATA_DIR")
        if data_dir_env:
            csv_path = Path(data_dir_env) / "crop_soil_suitability.csv"
        else:
            # Fallback: traverse up from src/llm_reasoner/llm_advisory_engine.py to root/data/
            base_dir = Path(__file__).resolve().parent.parent.parent
            csv_path = base_dir / "data" / "crop_soil_suitability.csv"
        
        if not csv_path.exists():
            logger.error(f"Suitability data file not found at {csv_path}")
            return []

        try:
            with open(csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append({
                        "district": row.get("district", ""),
                        "crop_name": row.get("crop_name", "")
                    })
        except Exception as e:
            logger.error(f"Error reading suitability CSV: {e}")
            
        return data

    def get_regional_crops(self, location: str) -> List[str]:
        """Returns a list of crops suitable for the given location (district)."""
        if not location:
            return []
        
        # Match location (case-insensitive) to district in CSV
        loc_lower = location.lower()
        crops = [
            row["crop_name"] for row in self.suitability_data 
            if loc_lower in row["district"].lower() or row["district"].lower() in loc_lower
        ]
        return list(set(crops)) # Unique crops

    def build_system_prompt(self) -> str:
        """
        Constructs the system prompt with strict constraints for the LLM.
        """
        return (
            "You are an expert agricultural advisory assistant. Your role is to translate "
            "technical crop recommendation and risk data into clear, actionable advice for farmers. "
            "STRICT CONSTRAINTS: "
            "1. You MUST NOT change crop rankings, modify any risk scores, or invent new risks. "
            "2. You MUST NOT generate or include any numerical values (scores, percentages) that are not present in the provided input JSON. "
            "3. You MUST use simple, farmer-friendly language. Avoid technical ML jargon. "
            "4. Your output must exactly follow the requested structure. "
            "5. CROSS-VALIDATION RULE: You will be provided with a 'Regional Suitability List' for the current location. "
            "If an ML-recommended crop is NOT in this list, you MUST explicitly state in the summary that "
            "this crop is 'not confidently suggested to grow in Bihar' due to lack of local suitability records."
            "Remember dont be technical because you are answering to the farmers so frame answers such that it can be understood by anyone"
        )

    def build_user_prompt(self, orchestrator_output: Dict[str, Any], regional_crops: List[str]) -> str:
        """
        Constructs a structured prompt based on the Orchestrator's JSON output.
        """
        json_data = json.dumps(orchestrator_output, indent=2)
        regional_list_str = ", ".join(regional_crops) if regional_crops else "No specific records found for this location."
        
        return f"""
Based on the following decision data from our agricultural systems:

```json
{json_data}
```

REGIONAL SUITABILITY LIST (Validated crops for this district):
[{regional_list_str}]

Generate a structured advisory report for the farmer.
You MUST format your output strictly with the following sections (use these exact headings):

1. Recommended Crops Summary
(Provide a clear summary. If a crop from the JSON is not in the Regional Suitability List, mark it as 'not confidently suggested to grow in Bihar'.)

2. Climate Risk Analysis
(Explain the specific climate risks: heat, drought, flood, and current weather context provided in the JSON.)

3. Disease Risk Advisory
(Explain the disease risk score and any related alerts.)

4. Preventive Measures
(Provide practical, actionable steps based on both soil parameters and weather conditions.)

5. Final Advisory Note
(A brief, encouraging closing statement summarizing the decision confidence.)

Remember: Do NOT add new crops, change scores, or invent facts outside the provided data.
"""

    def generate_advisory(self, orchestrator_output: Dict[str, Any]) -> str:
        """
        Invokes the LLM to generate the advisory text.
        """
        if not self.model:
            return "Error: ChatGroq client is not initialized. Check API key and package installation."
            
        try:
            location = orchestrator_output.get("location", "")
            regional_crops = self.get_regional_crops(location)
            
            messages = [
                SystemMessage(content=self.build_system_prompt()),
                HumanMessage(content=self.build_user_prompt(orchestrator_output, regional_crops))
            ]
            response = self.model.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating LLM advisory: {e}")
            return "Error: Unable to generate advisory at this time. Please review the raw decision data."
