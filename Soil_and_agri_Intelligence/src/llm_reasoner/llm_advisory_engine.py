import os
import json
import logging
from typing import Dict, Any

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
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initializes the LLM Advisory Engine using ChatGroq.
        
        Args:
            model_name: The Groq model to use.
        """
        load_dotenv()
        self.model_name = model_name
        
        if ChatGroq is None:
            logger.warning("Langchain Groq package not found. Please install with: pip install langchain-groq python-dotenv")
            self.model = None
        else:
            self.model = ChatGroq(model=self.model_name, temperature=0.2)

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
            "4. Your output must exactly follow the requested structure."
        )

    def build_user_prompt(self, orchestrator_output: Dict[str, Any]) -> str:
        """
        Constructs a structured prompt based on the Orchestrator's JSON output.
        """
        json_data = json.dumps(orchestrator_output, indent=2)
        
        return f"""
Based on the following decision data from our agricultural systems:

```json
{json_data}
```

Generate a structured advisory report for the farmer.
You MUST format your output strictly with the following sections (use these exact headings):

1. Recommended Crops Summary
(Provide a clear summary of the recommended crops based solely on the data provided.)

2. Climate Risk Analysis
(Explain the specific climate risks: heat, drought, flood, based only on the provided risk_summary. If a risk is low, mention that as well.)

3. Disease Risk Advisory
(Explain the disease risk score and any related alerts.)

4. Preventive Measures
(Provide practical, actionable steps a farmer can take to mitigate the identified risks.)

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
            messages = [
                SystemMessage(content=self.build_system_prompt()),
                HumanMessage(content=self.build_user_prompt(orchestrator_output))
            ]
            response = self.model.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating LLM advisory: {e}")
            return "Error: Unable to generate advisory at this time. Please review the raw decision data."
