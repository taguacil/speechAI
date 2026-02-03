"""Product information agent - CEAT product portfolio analysis."""

import json
import time
from dataclasses import dataclass
from typing import Any

from speechai.agents.base import AgentResult, BaseAgent

# CEAT Product Knowledge Base
CEAT_PRODUCTS = {
    "sportdrive_suv_calm": {
        "name": "SportDrive SUV CALM",
        "category": "Ultra-Premium SUV",
        "personas": ["entitled_evan"],
        "key_features": ["foam insert technology", "quiet cabin", "premium performance"],
    },
    "sportdrive": {
        "name": "SportDrive",
        "category": "Premium Performance",
        "personas": ["entitled_evan", "impatient_ashish"],
        "key_features": ["performance", "premium quality", "luxury"],
    },
    "crossdrive_at": {
        "name": "CrossDrive AT",
        "category": "All-Terrain",
        "personas": ["bindaas_bharat"],
        "key_features": ["durability", "rugged", "proven track record"],
    },
    "securadrive_suv": {
        "name": "SecuraDrive SUV",
        "category": "Highway/Urban SUV",
        "personas": ["pragmatic_purnima", "savvy_sarabh"],
        "key_features": ["wet grip", "dry grip", "hassle-free", "safety"],
    },
    "securadrive": {
        "name": "SecuraDrive",
        "category": "Standard Mass-Market",
        "personas": ["thorough_tushar", "pragmatic_purnima"],
        "key_features": ["safety", "widely adopted", "all-rounder"],
    },
    "energy_drive": {
        "name": "Energy Drive",
        "category": "EV-Specific",
        "personas": ["impatient_ashish", "entitled_evan"],
        "key_features": ["EV optimized", "innovative tech", "efficiency"],
    },
    "milaze_x5": {
        "name": "Milaze X5",
        "category": "Premium Economy",
        "personas": ["savvy_sarabh", "thorough_tushar"],
        "key_features": ["reliability", "convenience", "trustworthy"],
    },
    "milaze_x3": {
        "name": "Milaze X3",
        "category": "Economy / High-Mileage",
        "personas": ["savvy_sarabh", "bindaas_bharat"],
        "key_features": ["1 lakh km", "durability", "value"],
    },
}

UPSELL_TOOLKIT = {
    "calm_technology": {
        "name": "CALM Technology",
        "description": "Patented sound-absorbing foam insert",
        "target_personas": ["entitled_evan"],
        "trigger_keywords": ["noise", "quiet", "silent", "comfort", "cabin"],
    },
    "upsizing": {
        "name": "Upsizing",
        "description": "Larger diameter tyre for looks/handling",
        "target_personas": ["impatient_ashish"],
        "trigger_keywords": ["looks", "style", "bigger", "upgrade", "appearance"],
    },
    "run_flat": {
        "name": "Run-flat Tyres",
        "description": "Drive after puncture for limited distance",
        "target_personas": ["pragmatic_purnima"],
        "trigger_keywords": ["puncture", "flat", "stranded", "emergency", "safety"],
    },
    "lakh_km": {
        "name": "1 Lakh KM Tyre",
        "description": "Proven durability claim for Milaze X3",
        "target_personas": ["bindaas_bharat"],
        "trigger_keywords": ["long lasting", "durable", "mileage", "km", "kilometers"],
    },
}


@dataclass
class ProductResult:
    """Structured product result."""

    products_mentioned: list[str]
    ceat_products_matched: list[str]
    features_discussed: list[str]
    vehicle_type: str
    upsell_opportunities: list[str]
    pricing_concern: bool
    recommended_product: str


class ProductAgent(BaseAgent):
    """Agent for analyzing CEAT product mentions and upsell opportunities."""

    name = "product"
    default_model = "gpt-5-mini"  # Reliable JSON output

    async def analyze(self, text: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Analyze product mentions in customer speech.

        Args:
            text: Customer's speech text.
            context: Optional conversation context.

        Returns:
            AgentResult containing ProductResult data.
        """
        start = time.perf_counter()

        if not text.strip():
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=self._empty_result(),
                latency_ms=0,
            )

        prompts = self.prompts.get("product", {})
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
            "products_mentioned": [],
            "ceat_products_matched": [],
            "features_discussed": [],
            "vehicle_type": "",
            "upsell_opportunities": [],
            "pricing_concern": False,
            "recommended_product": "",
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

            return {
                "products_mentioned": data.get("products_mentioned", [])[:5],
                "ceat_products_matched": data.get("ceat_products_matched", [])[:3],
                "features_discussed": data.get("features_discussed", [])[:5],
                "vehicle_type": str(data.get("vehicle_type", ""))[:50],
                "upsell_opportunities": data.get("upsell_opportunities", [])[:3],
                "pricing_concern": bool(data.get("pricing_concern", False)),
                "recommended_product": str(data.get("recommended_product", ""))[:50],
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[product] Parse error: {e}")
            return None

    @staticmethod
    def to_result(data: dict[str, Any]) -> ProductResult:
        """Convert data dict to typed ProductResult."""
        return ProductResult(
            products_mentioned=data.get("products_mentioned", []),
            ceat_products_matched=data.get("ceat_products_matched", []),
            features_discussed=data.get("features_discussed", []),
            vehicle_type=data.get("vehicle_type", ""),
            upsell_opportunities=data.get("upsell_opportunities", []),
            pricing_concern=data.get("pricing_concern", False),
            recommended_product=data.get("recommended_product", ""),
        )
