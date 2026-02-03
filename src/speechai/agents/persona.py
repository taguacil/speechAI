"""Persona inference agent - CEAT customer personas."""

import json
import time
from dataclasses import dataclass
from typing import Any

from speechai.agents.base import AgentResult, BaseAgent

# CEAT Customer Personas
PERSONAS = {
    "entitled_evan": {
        "name": "Entitled Evan",
        "segment": "Premium Seekers",
        "needs": ["comfort", "safety", "noiseless ride", "premium looks", "new technology"],
        "buying_behavior": ["seeks exclusive experiences", "attracted to upgrades", "delegates research"],
        "products": ["SportDrive SUV CALM", "SportDrive", "Energy Drive"],
        "triggers": ["luxury", "premium", "best", "exclusive", "BMW", "Mercedes", "Audi", "comfort", "quiet"],
    },
    "impatient_ashish": {
        "name": "Impatient Ashish",
        "segment": "Premium Seekers",
        "needs": ["performance", "looks", "technology", "quick decisions"],
        "buying_behavior": ["wants latest tech", "impulsive", "style-conscious"],
        "products": ["SportDrive", "Energy Drive"],
        "triggers": ["fast", "quick", "latest", "new", "style", "looks", "upgrade", "EV", "electric"],
    },
    "bindaas_bharat": {
        "name": "Bindaas Bharat",
        "segment": "Value & Trust Seekers",
        "needs": ["durability", "proven track record", "value for money"],
        "buying_behavior": ["dislikes tall claims", "trusts experience", "no-nonsense"],
        "products": ["Milaze X3", "CrossDrive AT"],
        "triggers": ["lakh km", "durable", "long lasting", "proven", "reliable", "value", "Indian"],
    },
    "thorough_tushar": {
        "name": "Thorough Tushar",
        "segment": "Value & Trust Seekers",
        "needs": ["warranty", "safety", "widely adopted brands"],
        "buying_behavior": ["researches meticulously", "reads reviews", "compares options"],
        "products": ["SecuraDrive", "Milaze X5"],
        "triggers": ["warranty", "reviews", "compare", "safety rating", "which is better", "research"],
    },
    "pragmatic_purnima": {
        "name": "Pragmatic Purnima",
        "segment": "Efficiency & Convenience Seekers",
        "needs": ["puncture resistance", "monsoon safety", "hassle-free"],
        "buying_behavior": ["wants smart choices", "problem solver", "practical"],
        "products": ["SecuraDrive SUV", "SecuraDrive"],
        "triggers": ["puncture", "monsoon", "rain", "safety", "hassle", "easy", "convenient"],
    },
    "savvy_sarabh": {
        "name": "Savvy Sarabh",
        "segment": "Efficiency & Convenience Seekers",
        "needs": ["convenience", "trustworthy brand", "efficiency"],
        "buying_behavior": ["values trust", "brand conscious", "seeks convenience"],
        "products": ["SecuraDrive SUV", "Milaze X5", "Milaze X3"],
        "triggers": ["trust", "brand", "convenient", "easy", "efficient", "recommend"],
    },
}

PERSONA_SEGMENTS = {
    "premium_seekers": {
        "name": "Premium Seekers",
        "personas": ["entitled_evan", "impatient_ashish"],
        "pitch_style": "Emphasize exclusivity, technology, premium experience",
    },
    "value_trust_seekers": {
        "name": "Value & Trust Seekers",
        "personas": ["bindaas_bharat", "thorough_tushar"],
        "pitch_style": "Emphasize proven track record, durability, warranty, reviews",
    },
    "efficiency_convenience_seekers": {
        "name": "Efficiency & Convenience Seekers",
        "personas": ["pragmatic_purnima", "savvy_sarabh"],
        "pitch_style": "Emphasize hassle-free experience, safety, convenience",
    },
}


@dataclass
class PersonaResult:
    """Structured persona result."""

    persona_id: str
    persona_name: str
    segment: str
    confidence: float
    detected_triggers: list[str]
    recommended_products: list[str]
    pitch_style: str


class PersonaAgent(BaseAgent):
    """Agent for inferring CEAT customer persona from speech."""

    name = "persona"
    default_model = "gpt-5-mini"  # Reliable JSON output

    async def analyze(self, text: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Analyze customer speech to infer persona.

        Args:
            text: Customer's speech text.
            context: Optional conversation context.

        Returns:
            AgentResult containing PersonaResult data.
        """
        start = time.perf_counter()

        if not text.strip():
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=self._empty_result(),
                latency_ms=0,
            )

        prompts = self.prompts.get("persona", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "")

        user_prompt = self._format_prompt(user_template, text=text)
        response = await self._call_llm(system_prompt, user_prompt)

        latency_ms = (time.perf_counter() - start) * 1000
        result = self._parse_response(response)

        return AgentResult(
            agent_name=self.name,
            success=result is not None,
            data=result or self._empty_result(),
            latency_ms=latency_ms,
        )

    def _empty_result(self) -> dict[str, Any]:
        """Return empty result structure."""
        return {
            "persona_id": "",
            "persona_name": "",
            "segment": "",
            "confidence": 0.0,
            "detected_triggers": [],
            "recommended_products": [],
            "pitch_style": "",
        }

    def _parse_response(self, response: str | None) -> dict[str, Any] | None:
        """Parse LLM JSON response."""
        if not response:
            return None

        try:
            content = response
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            persona_id = data.get("persona_id", "")
            persona_info = PERSONAS.get(persona_id, {})

            return {
                "persona_id": persona_id,
                "persona_name": persona_info.get("name", data.get("persona_name", "")),
                "segment": persona_info.get("segment", data.get("segment", "")),
                "confidence": min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
                "detected_triggers": data.get("detected_triggers", [])[:5],
                "recommended_products": persona_info.get("products", data.get("recommended_products", []))[:3],
                "pitch_style": data.get("pitch_style", "")[:100],
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[persona] Parse error: {e}")
            return None

    @staticmethod
    def to_result(data: dict[str, Any]) -> PersonaResult:
        """Convert data dict to typed PersonaResult."""
        return PersonaResult(
            persona_id=data.get("persona_id", ""),
            persona_name=data.get("persona_name", ""),
            segment=data.get("segment", ""),
            confidence=data.get("confidence", 0.0),
            detected_triggers=data.get("detected_triggers", []),
            recommended_products=data.get("recommended_products", []),
            pitch_style=data.get("pitch_style", ""),
        )
