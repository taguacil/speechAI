"""Competition analysis agent - Indian tire market intelligence."""

import json
import time
from dataclasses import dataclass
from typing import Any

from speechai.agents.base import AgentResult, BaseAgent

# Competitor Intelligence Database
COMPETITORS = {
    "mrf": {
        "name": "MRF",
        "full_name": "MRF Limited (Madras Rubber Factory)",
        "market_share": "29-30%",
        "ranking": "#1 in India",
        "positioning": "Market leader, strongest brand",
        "products": ["ZVTV", "ZLX", "Perfinza", "Wanderer"],
        "strengths": ["Largest market share", "Strongest brand reputation", "Extensive dealer network", "Cricket sponsorships"],
        "weaknesses": ["Less flexible to market trends", "Noisier ride (rugged build)"],
        "price_position": "Mid-range to Premium",
        "counter_positioning": "CEAT offers similar durability with better comfort and noise reduction through CALM technology",
    },
    "apollo": {
        "name": "Apollo",
        "full_name": "Apollo Tyres Ltd.",
        "market_share": "20%",
        "ranking": "#2 in India",
        "positioning": "Comfort leader, dual-brand strategy",
        "products": ["Amazer", "Alnac", "Apterra", "Amperion (EV)", "Vredestein"],
        "strengths": ["Best natural noise comfort", "Strong OEM presence", "International presence", "EV-specific range"],
        "weaknesses": ["Developing premium brand image", "Limited in new car models"],
        "price_position": "Mid-range to Premium",
        "counter_positioning": "CEAT's CALM technology offers superior noise reduction, plus run-flat capability Apollo doesn't have",
    },
    "jk_tyre": {
        "name": "JK Tyre",
        "full_name": "JK Tyre & Industries Ltd.",
        "market_share": "19%",
        "ranking": "#3-4 in India",
        "positioning": "Value leader, radial technology pioneer",
        "products": ["Levitas Ultra", "UX Royale", "Ranger", "Blazze"],
        "strengths": ["Best value for money", "Pioneer of radial technology", "Smart Tyre TPMS", "Strong motorsports presence"],
        "weaknesses": ["Lower premium perception"],
        "price_position": "Mid-range to Premium",
        "counter_positioning": "CEAT offers better quality credentials (Deming Prize winner) with competitive pricing",
    },
    "bridgestone": {
        "name": "Bridgestone",
        "full_name": "Bridgestone India Private Limited",
        "market_share": "20% (replacement)",
        "ranking": "#2-3 in India",
        "positioning": "Global technology leader",
        "products": ["Turanza", "Potenza", "Alenza", "Sturdo", "Dueler", "Ecopia"],
        "strengths": ["World's largest tire manufacturer", "Global R&D access", "Full price spectrum", "B-Silent technology"],
        "weaknesses": ["Perceived as premium-priced", "Only 2 plants in India"],
        "price_position": "Budget to Premium (full spectrum)",
        "counter_positioning": "CEAT is Made in India with global quality standards, Deming Prize winner - same quality at better value",
    },
    "michelin": {
        "name": "Michelin",
        "full_name": "Michelin",
        "market_share": "Not disclosed",
        "ranking": "Premium segment player",
        "positioning": "Ultra-premium, global leader",
        "products": ["Primacy 4", "Pilot Sport 4", "Latitude Tour"],
        "strengths": ["Premium brand image", "Excellent wet grip", "Long tread life"],
        "weaknesses": ["Very high pricing", "Limited India manufacturing"],
        "price_position": "Premium to Luxury",
        "counter_positioning": "CEAT SportDrive offers comparable performance at significantly lower price, with local warranty support",
    },
    "goodyear": {
        "name": "Goodyear",
        "full_name": "Goodyear India Limited",
        "market_share": "Not disclosed",
        "ranking": "Lower tier",
        "positioning": "Technology innovator",
        "products": ["Assurance TripleMax 2", "Assurance ComfortTred", "Assurance MaxGuard", "DuraPlus 2"],
        "strengths": ["HydroTred technology", "ANX noise tech", "Pioneer in tubeless radial"],
        "weaknesses": ["Less dominant market position", "Focus on trading vs manufacturing"],
        "price_position": "Mid-range to Premium",
        "counter_positioning": "CEAT offers better availability and local manufacturing with comparable technology",
    },
}

# CEAT's unique differentiators for counter-positioning
CEAT_DIFFERENTIATORS = {
    "run_flat": {
        "claim": "First and only Indian manufacturer of run-flat tyres",
        "details": "SportDrive RFT - can run 80km at 80km/h after puncture",
        "competitors_with_feature": [],  # No Indian competitor has this
    },
    "calm_technology": {
        "claim": "Proprietary CALM technology for noise reduction",
        "details": "8-10dB noise reduction with foam inserts, ZR-rated for 300+ km/h",
        "competitors_with_feature": ["Bridgestone (B-Silent)", "Pirelli (PNCS)"],
    },
    "deming_prize": {
        "claim": "First tire company globally to win the Deming Prize (2023)",
        "details": "Highest quality recognition in manufacturing",
        "competitors_with_feature": [],
    },
    "sustainability": {
        "claim": "India's first sustainable road-ready passenger car tyres",
        "details": "SecuraDrive CIRCL launched August 2025",
        "competitors_with_feature": [],
    },
    "indian_heritage": {
        "claim": "Proud Indian brand, part of RPG Group",
        "details": "6 manufacturing plants across India, 48 million tyres/year",
        "competitors_with_feature": ["MRF", "Apollo", "JK Tyre"],
    },
}

# Price positioning by segment
PRICE_POSITIONING = {
    "12-13_inch": {
        "ceat": "₹2,500-4,500 (Milaze, SecuraDrive)",
        "mrf": "₹3,000-5,000 (ZVTV, Zapper)",
        "apollo": "₹2,800-4,800 (Amazer 3G)",
        "jk_tyre": "₹2,400-4,200 (Ultima, Vectra) - Most affordable",
    },
    "14-16_inch": {
        "ceat": "₹3,500-7,500 (SportDrive, Gripp)",
        "mrf": "₹4,000-8,500 (ZLX, Wanderer)",
        "apollo": "₹3,800-7,800 (Alnac 4G) - Quietest",
        "jk_tyre": "₹3,200-7,000 (UX Royale)",
    },
    "17_plus_inch": {
        "ceat": "₹5,500-30,000 (SportDrive, SportDrive CALM)",
        "mrf": "₹6,000-15,000 (Perfinza)",
        "apollo": "₹5,800-13,000 (Aspire XP)",
        "bridgestone": "₹6,000-25,000 (Turanza, Potenza)",
    },
}


@dataclass
class CompetitionResult:
    """Structured competition result."""

    competitors_mentioned: list[str]
    competitor_products: list[str]
    competitive_concern: bool
    switching_intent: bool
    price_comparison: bool
    counter_positioning: str
    ceat_differentiator: str


class CompetitionAgent(BaseAgent):
    """Agent for analyzing competitor mentions and competitive positioning."""

    name = "competition"
    default_model = "gpt-5-mini"  # Reliable JSON output

    async def analyze(self, text: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Analyze competitor mentions in customer speech.

        Args:
            text: Customer's speech text.
            context: Optional conversation context.

        Returns:
            AgentResult containing CompetitionResult data.
        """
        start = time.perf_counter()

        if not text.strip():
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=self._empty_result(),
                latency_ms=0,
            )

        prompts = self.prompts.get("competition", {})
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
            "competitors_mentioned": [],
            "competitor_products": [],
            "competitive_concern": False,
            "switching_intent": False,
            "price_comparison": False,
            "counter_positioning": "",
            "ceat_differentiator": "",
        }

    def _parse_response(self, response: str | None) -> dict[str, Any] | None:
        """Parse LLM JSON response and enrich with competitive intelligence."""
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

            # Extract competitors mentioned
            competitors = data.get("competitors_mentioned", [])[:3]

            # Get counter-positioning for first competitor mentioned
            counter = ""
            differentiator = ""
            if competitors:
                first_competitor = competitors[0].lower().replace(" ", "_").replace("-", "_")
                competitor_info = COMPETITORS.get(first_competitor, {})
                counter = competitor_info.get("counter_positioning", "")

                # Select best differentiator based on context
                if data.get("price_comparison"):
                    differentiator = "Deming Prize quality at competitive price"
                elif "noise" in str(data).lower() or "quiet" in str(data).lower():
                    differentiator = CEAT_DIFFERENTIATORS["calm_technology"]["claim"]
                elif "puncture" in str(data).lower() or "flat" in str(data).lower():
                    differentiator = CEAT_DIFFERENTIATORS["run_flat"]["claim"]
                else:
                    differentiator = CEAT_DIFFERENTIATORS["deming_prize"]["claim"]

            return {
                "competitors_mentioned": competitors,
                "competitor_products": data.get("competitor_products", [])[:3],
                "competitive_concern": bool(data.get("competitive_concern", False)),
                "switching_intent": bool(data.get("switching_intent", False)),
                "price_comparison": bool(data.get("price_comparison", False)),
                "counter_positioning": counter[:150] if counter else data.get("counter_positioning", "")[:150],
                "ceat_differentiator": differentiator[:100],
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[competition] Parse error: {e}")
            return None

    @staticmethod
    def to_result(data: dict[str, Any]) -> CompetitionResult:
        """Convert data dict to typed CompetitionResult."""
        return CompetitionResult(
            competitors_mentioned=data.get("competitors_mentioned", []),
            competitor_products=data.get("competitor_products", []),
            competitive_concern=data.get("competitive_concern", False),
            switching_intent=data.get("switching_intent", False),
            price_comparison=data.get("price_comparison", False),
            counter_positioning=data.get("counter_positioning", ""),
            ceat_differentiator=data.get("ceat_differentiator", ""),
        )
