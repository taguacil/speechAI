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
    "sportdrive_calm": {
        "name": "SportDrive Calm",
        "category": "Premium Comfort",
        "personas": ["entitled_evan", "thorough_tushar"],
        "key_features": ["low noise", "smooth ride", "comfort", "quiet cabin", "highway comfort"],
        "main_strength": "Comfort & low noise",
        "typical_use": "City sedans, hatchbacks, premium vehicles",
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

# Car Type to Product Mapping with Size Information
CAR_TYPE_MAPPING = {
    "entry_hatchback": {
        "car_examples": ["Alto", "Alto K10", "Wagon R", "Celerio", "Santro", "Kwid"],
        "recommended_products": ["milaze_x3", "milaze_x5", "securadrive"],
        "product_notes": {
            "milaze_x3": "Lowest cost, long life, city use",
            "milaze_x5": "Better grip & stability than X3",
            "securadrive": "Better braking, wet grip, comfort",
        },
        "standard_size": "165/80 R14",
        "upsize_option": "165/70 R14",
        "upsize_benefits": ["Better grip & braking", "More stable at speed"],
    },
    "premium_hatchback": {
        "car_examples": ["Swift", "Baleno", "i20", "Altroz", "Polo", "Jazz", "Glanza"],
        "recommended_products": ["securadrive", "sportdrive", "sportdrive_calm"],
        "product_notes": {
            "securadrive": "Balanced daily comfort",
            "sportdrive": "Sporty handling, strong braking",
            "sportdrive_calm": "Low noise, smooth ride",
        },
        "standard_size": "175/65 R15",
        "upsize_option": "195/55 R16",
        "upsize_benefits": ["Better cornering", "Sportier look", "More confidence"],
    },
    "compact_sedan": {
        "car_examples": ["City", "Verna", "Ciaz", "Slavia", "Virtus", "Rapid", "Yaris"],
        "recommended_products": ["securadrive", "sportdrive", "sportdrive_calm"],
        "product_notes": {
            "securadrive": "Comfortable, fuel-friendly",
            "sportdrive": "Better steering response",
            "sportdrive_calm": "Quiet cabin, smooth highways",
        },
        "standard_size": "185/65 R15",
        "upsize_option": "195/55 R16",
        "upsize_benefits": ["Better highway stability", "Improved braking"],
    },
    "mid_size_sedan": {
        "car_examples": ["Octavia", "Corolla Altis", "Elantra", "Camry"],
        "recommended_products": ["sportdrive", "sportdrive_calm"],
        "product_notes": {
            "sportdrive": "High-speed grip & control",
            "sportdrive_calm": "Premium comfort, low noise",
        },
        "standard_size": "205/55 R16",
        "upsize_option": "215/55 R17",
        "upsize_benefits": ["Smoother high-speed drive", "Premium comfort"],
    },
    "compact_suv": {
        "car_examples": ["Brezza", "Venue", "Sonet", "XUV300", "Taigun", "Kushaq"],
        "recommended_products": ["securadrive_suv", "sportdrive_suv"],
        "product_notes": {
            "securadrive_suv": "Built for SUV weight, comfort",
            "sportdrive_suv": "Better grip & cornering",
        },
        "standard_size": "205/60 R16",
        "upsize_option": "215/55 R17",
        "upsize_benefits": ["Better SUV stability", "Improved road grip"],
    },
    "mid_size_suv": {
        "car_examples": ["Creta", "Seltos", "Hector", "Harrier", "Compass", "Safari"],
        "recommended_products": ["securadrive_suv", "sportdrive_suv", "sportdrive_calm"],
        "product_notes": {
            "securadrive_suv": "Daily driving comfort",
            "sportdrive_suv": "Sporty control",
            "sportdrive_calm": "Quiet premium ride",
        },
        "standard_size": "215/60 R17",
        "upsize_option": "215/55 R18",
        "upsize_benefits": ["Better cornering", "More planted feel"],
    },
    "large_premium_suv": {
        "car_examples": ["Fortuner", "Endeavour", "Prado", "X5", "GLE", "Q7", "Range Rover"],
        "recommended_products": ["sportdrive_calm", "sportdrive_suv", "crossdrive_at"],
        "product_notes": {
            "sportdrive_calm": "Luxury silence & comfort",
            "sportdrive_suv": "Performance handling",
            "crossdrive_at": "Strong sidewalls, rough roads",
        },
        "standard_size": "235/55 R19",
        "upsize_options": ["255/50 R19", "275/45 R20"],
        "upsize_benefits": ["Stronger grip", "Luxury feel", "Better road presence"],
    },
    "offroad_adventure_suv": {
        "car_examples": ["Thar", "Scorpio", "Jimny", "Fortuner 4x4"],
        "recommended_products": ["crossdrive_at", "sportdrive_suv"],
        "product_notes": {
            "crossdrive_at": "All-terrain traction, durability",
            "sportdrive_suv": "Highway comfort + strength",
        },
        "standard_size": "245/70 R16",
        "upsize_option": "Wider AT sizes",
        "upsize_benefits": ["Better traction on sand/mud", "Stronger sidewall"],
    },
    "electric_vehicle": {
        "car_examples": ["Nexon EV", "ZS EV", "Kona EV", "Atto 3"],
        "recommended_products": ["energy_drive"],
        "product_notes": {
            "energy_drive": "Low noise, low rolling resistance, higher load capacity",
        },
        "standard_size": "215/60 R16",
        "upsize_option": "215/55 R17",
        "upsize_benefits": ["Better grip without noise", "Minimal range impact"],
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

        context = context or {}
        role = context.get("role", "Customer")

        prompts = self.prompts.get("product", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "")

        user_prompt = self._format_prompt(user_template, text=text, role=role)
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
