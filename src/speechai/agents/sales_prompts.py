"""Sales prompts agent - contextual scripts and objection handlers for CEAT sales."""

import json
import time
from dataclasses import dataclass
from typing import Any

from speechai.agents.base import AgentResult, BaseAgent

# Upselling Scripts
UPSELL_SCRIPTS = {
    "2_to_4_tyres": {
        "trigger_keywords": ["two tyres", "2 tyres", "only two", "just two", "pair"],
        "arguments": {
            "safety": "If you put two new tyres on front but leave old ones on back, your car becomes unbalanced. In monsoon or emergency braking, the back could 'fishtail'. With four new tyres, braking is uniform - much safer for your family.",
            "suspension": "It's like wearing one new shoe and one old chappal - you'll walk funny. Mismatched tread depths stress your suspension and alignment. All four ensures smooth handling.",
            "savings": "Buying two now and two later costs more. You can't properly rotate tyres if half are worn. We have a 'Set of 4' offer with better warranty and free alignment.",
        },
        "closing": "Instead of fixing a problem today, let's set your car up for the next 40,000-50,000 km. Small extra investment now for safety and savings later.",
    },
    "4_to_5_tyres": {
        "trigger_keywords": ["four tyres", "4 tyres", "set of four", "all four"],
        "arguments": {
            "rotation": "If you buy five today, include the spare in regular rotation. This extends life of entire set by 20% - you won't visit us again for much longer.",
            "aging": "Even unused spares lose grip over time - rubber becomes brittle. In a midnight puncture emergency, you don't want to rely on an old tyre.",
        },
        "closing": "Since you're getting four, I can bundle the fifth. All five wheels with same tread pattern for balanced handling. Should I add that fifth tyre?",
    },
    "15_to_17_inch": {
        "trigger_keywords": ["15 inch", "15-inch", "upgrade", "bigger tyres", "larger size"],
        "arguments": {
            "stability": "With 17-inch, you have shorter sidewall - less 'tyre sway' when changing lanes. Car feels planted, like it's on rails.",
            "braking": "17-inch has wider contact patch - more rubber on road. Stopping distance improves, steering more responsive in turns.",
            "appearance": "15-inch can look 'small' in modern wheel arches. 17s fill that gap perfectly - premium, sporty stance.",
        },
        "closing": "It's the single most effective way to make your car feel brand new. Shall I check compatibility and pricing for 17-inch performance set?",
    },
    "run_flat": {
        "trigger_keywords": ["puncture", "flat tyre", "emergency", "stranded", "midnight", "highway"],
        "arguments": {
            "safety": "With Run-Flats, if you get puncture at 11 PM on lonely highway, you don't stop. Keep driving 80km at 80km/h to reach home or service center.",
            "convenience": "No changing tyres roadside, no waiting for help, no safety risk for family.",
        },
        "closing": "It's the ultimate safety net. CEAT is the only Indian manufacturer of run-flat tyres - global technology with Indian durability.",
    },
    "calm_technology": {
        "trigger_keywords": ["noise", "quiet", "silent", "cabin noise", "road noise", "comfort"],
        "arguments": {
            "technology": "CEAT CALM technology uses foam inserts that absorb vibrations - 8-10dB noise reduction. Your cabin becomes incredibly quiet.",
            "ev_benefit": "Especially important for EVs where there's no engine noise to mask tyre sound.",
        },
        "closing": "SportDrive CALM is ZR-rated for 300+ km/h with luxury-car quietness. Perfect for your vehicle.",
    },
}

# Objection Handling Scripts
OBJECTION_HANDLERS = {
    "already_purchased_competitor": {
        "trigger_keywords": ["already bought", "purchased", "got from", "bought mrf", "bought apollo", "bought bridgestone"],
        "response": "I understand. May I ask what influenced your decision? This helps us improve. Also, we'd love to earn your business next time - can I send you information about our loyalty program for your next tyre purchase?",
        "follow_up": "Add to 6-month follow-up list with competitor notes",
    },
    "price_question": {
        "trigger_keywords": ["price", "cost", "how much", "kitna", "rate", "expensive", "budget"],
        "response": "Great question! While final pricing is confirmed by dealers based on location, typical range for this tyre is ₹{price_low} to ₹{price_high}. Your nearest dealer can give exact quote. Would you like me to connect you directly?",
        "note": "Always provide a range, never say 'dealer will confirm' without context",
    },
    "product_unavailable": {
        "trigger_keywords": ["not available", "out of stock", "don't have", "no stock"],
        "response": "I understand that's frustrating. While we don't currently stock that exact size, we have {alternative} which many customers use for {vehicle}. Alternatively, I can check when that size will be available and call you back. Which would you prefer?",
    },
    "competitor_comparison": {
        "mrf": {
            "trigger": ["mrf", "madras rubber"],
            "response": "MRF is a strong brand with great durability. However, CEAT offers similar longevity with better comfort through our CALM technology. We're also the first tyre company globally to win the Deming Prize for quality.",
        },
        "apollo": {
            "trigger": ["apollo"],
            "response": "Apollo makes good tyres. What sets CEAT apart is our run-flat technology - we're the only Indian manufacturer. Plus our CALM foam technology for noise reduction that Apollo doesn't offer.",
        },
        "bridgestone": {
            "trigger": ["bridgestone"],
            "response": "Bridgestone is a premium global brand. CEAT offers the same high-speed safety ratings and wet-braking performance, but engineered specifically for Indian roads - our compound handles potholes and heat better. And you save 15-20% with local warranty support.",
        },
        "michelin": {
            "trigger": ["michelin"],
            "response": "Michelin is ultra-premium. CEAT SportDrive offers comparable performance at significantly lower price, with 5-year warranty and 4,000+ dealers across India for easy service.",
        },
    },
    "timing_inconvenient": {
        "trigger_keywords": ["busy", "not now", "later", "call back", "bad time"],
        "response": "I understand. When would be a better time to call? I'll make a note and reach out then.",
        "note": "Respect customer time - short calls are better than forced long ones",
    },
    "need_to_think": {
        "trigger_keywords": ["think about it", "decide later", "not sure", "let me check"],
        "response": "Of course, it's an important decision. Can I send you a comparison on WhatsApp so you have all the information? And may I follow up in a day or two?",
    },
    "dealer_issues": {
        "trigger_keywords": ["dealer closed", "couldn't reach", "no response", "bad experience"],
        "response": "I sincerely apologize for that experience. Let me connect you with our priority support and find an alternative dealer. We also have our toll-free number 1800-267-1213 available 24/7.",
    },
}

# Call Structure Guidelines
CALL_GUIDELINES = {
    "new_lead": {
        "duration": "3-5 minutes",
        "structure": ["Qualify needs", "Recommend product", "Schedule dealer visit"],
        "opening": "Hello {name}, this is {agent} from CEAT Tyres. Do you have 3-4 minutes to discuss your tyre inquiry?",
    },
    "post_purchase": {
        "duration": "2-3 minutes",
        "structure": ["Confirm purchase", "Register warranty", "Thank customer"],
        "opening": "Hello {name}, I'm calling to thank you for choosing CEAT and help you register your warranty.",
    },
    "follow_up": {
        "duration": "4-6 minutes",
        "structure": ["Status check", "Address concerns", "Close or reschedule"],
        "opening": "Hello {name}, I'm following up on your interest in CEAT tyres. Have you had a chance to visit our dealer?",
    },
}

# Quick Response Templates
QUICK_RESPONSES = {
    "greeting": "Namaste! Thank you for calling CEAT Tyres.",
    "vehicle_check": "I see you're looking at tyres for your {vehicle}. Those are great cars - let me find the perfect match.",
    "warranty_value": "With CEAT, you get up to 5 years warranty plus our 6-month additional coverage. If anything goes wrong, we've got you covered.",
    "dealer_intro": "Your nearest dealer is {dealer_name} at {location}. They have your size in stock and can fit today.",
    "closing_soft": "Shall I have our dealer call you to confirm the appointment?",
    "closing_strong": "Based on everything we discussed, the {product} is perfect for your {vehicle}. Can I book your dealer visit for tomorrow?",
    "graceful_exit": "Thank you for considering CEAT. Please keep our toll-free number 1800-267-1213 for any future needs.",
}


@dataclass
class SalesPromptsResult:
    """Structured sales prompts result."""

    recommended_response: str
    upsell_opportunity: str
    upsell_script: str
    objection_detected: str
    objection_handler: str
    next_action: str
    call_stage: str


class SalesPromptsAgent(BaseAgent):
    """Agent for generating contextual sales prompts and objection handlers."""

    name = "sales_prompts"
    default_model = "claude-haiku-4-5"  # Best tone for customer-facing scripts

    async def analyze(self, text: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Generate sales prompts based on customer speech.

        Args:
            text: Customer's speech text.
            context: Optional conversation context.

        Returns:
            AgentResult containing SalesPromptsResult data.
        """
        start = time.perf_counter()

        if not text.strip():
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=self._empty_result(),
                latency_ms=0,
            )

        prompts = self.prompts.get("sales_prompts", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "")

        user_prompt = self._format_prompt(user_template, text=text)
        response = await self._call_llm(system_prompt, user_prompt)

        latency_ms = (time.perf_counter() - start) * 1000
        result = self._parse_response(response, text)

        return AgentResult(
            agent_name=self.name,
            success=result is not None,
            data=result or self._empty_result(),
            latency_ms=latency_ms,
        )

    def _empty_result(self) -> dict[str, Any]:
        """Return empty result structure."""
        return {
            "recommended_response": "",
            "upsell_opportunity": "",
            "upsell_script": "",
            "objection_detected": "",
            "objection_handler": "",
            "next_action": "",
            "call_stage": "discovery",
        }

    def _detect_upsell_opportunity(self, text: str) -> tuple[str, str]:
        """Detect upsell opportunity from customer text."""
        text_lower = text.lower()
        for upsell_key, upsell_data in UPSELL_SCRIPTS.items():
            for keyword in upsell_data["trigger_keywords"]:
                if keyword in text_lower:
                    # Get the best argument
                    arguments = upsell_data["arguments"]
                    first_arg_key = list(arguments.keys())[0]
                    return upsell_key, arguments[first_arg_key]
        return "", ""

    def _detect_objection(self, text: str) -> tuple[str, str]:
        """Detect objection from customer text."""
        text_lower = text.lower()

        # Check competitor mentions first
        competitors = OBJECTION_HANDLERS.get("competitor_comparison", {})
        for comp_key, comp_data in competitors.items():
            if comp_key == "trigger":
                continue
            for trigger in comp_data.get("trigger", []):
                if trigger in text_lower:
                    return f"competitor_{comp_key}", comp_data["response"]

        # Check other objections
        for obj_key, obj_data in OBJECTION_HANDLERS.items():
            if obj_key == "competitor_comparison":
                continue
            for keyword in obj_data.get("trigger_keywords", []):
                if keyword in text_lower:
                    return obj_key, obj_data["response"]

        return "", ""

    def _parse_response(self, response: str | None, original_text: str) -> dict[str, Any] | None:
        """Parse LLM JSON response and enrich with embedded scripts."""
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

            # Detect upsell opportunity from original text
            upsell_key, upsell_script = self._detect_upsell_opportunity(original_text)

            # Detect objection from original text
            objection_key, objection_handler = self._detect_objection(original_text)

            # Use detected values or fall back to LLM response
            return {
                "recommended_response": str(data.get("recommended_response", ""))[:200],
                "upsell_opportunity": upsell_key or data.get("upsell_opportunity", ""),
                "upsell_script": upsell_script[:200] if upsell_script else data.get("upsell_script", "")[:200],
                "objection_detected": objection_key or data.get("objection_detected", ""),
                "objection_handler": objection_handler[:200] if objection_handler else data.get("objection_handler", "")[:200],
                "next_action": str(data.get("next_action", ""))[:100],
                "call_stage": data.get("call_stage", "discovery"),
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[sales_prompts] Parse error: {e}")
            # Still try to detect from original text even if LLM fails
            upsell_key, upsell_script = self._detect_upsell_opportunity(original_text)
            objection_key, objection_handler = self._detect_objection(original_text)

            if upsell_key or objection_key:
                return {
                    "recommended_response": "",
                    "upsell_opportunity": upsell_key,
                    "upsell_script": upsell_script[:200],
                    "objection_detected": objection_key,
                    "objection_handler": objection_handler[:200],
                    "next_action": "",
                    "call_stage": "discovery",
                }
            return None

    @staticmethod
    def to_result(data: dict[str, Any]) -> SalesPromptsResult:
        """Convert data dict to typed SalesPromptsResult."""
        return SalesPromptsResult(
            recommended_response=data.get("recommended_response", ""),
            upsell_opportunity=data.get("upsell_opportunity", ""),
            upsell_script=data.get("upsell_script", ""),
            objection_detected=data.get("objection_detected", ""),
            objection_handler=data.get("objection_handler", ""),
            next_action=data.get("next_action", ""),
            call_stage=data.get("call_stage", "discovery"),
        )
