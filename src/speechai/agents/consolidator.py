"""Consolidator that combines agent outputs into actionable suggestions."""

import json
import time
from dataclasses import dataclass, field
from typing import Any

from speechai.agents.base import AgentResult, BaseAgent


@dataclass
class Suggestion:
    """A single suggestion for the sales rep."""

    text: str


@dataclass
class ConsolidatedOutput:
    """Final output shown to sales rep - includes all agent insights."""

    # Core suggestions
    suggestions: list[Suggestion]

    # Sentiment analysis
    sentiment: str
    confidence: float
    signals: list[str]

    # Persona inference
    persona_id: str = ""
    persona_name: str = ""
    persona_segment: str = ""

    # Product analysis
    products_mentioned: list[str] = field(default_factory=list)
    upsell_opportunities: list[str] = field(default_factory=list)
    recommended_product: str = ""

    # Competition analysis
    competitors_mentioned: list[str] = field(default_factory=list)
    counter_positioning: str = ""

    # Sales prompts
    objection_detected: str = ""
    objection_handler: str = ""
    upsell_script: str = ""
    call_stage: str = "discovery"

    # Metadata
    total_latency_ms: float = 0.0


class Consolidator(BaseAgent):
    """Consolidates parallel agent outputs into actionable suggestions.

    This runs AFTER all parallel agents complete and combines their
    results into 2-3 bullet suggestions for the sales rep.
    """

    name = "consolidator"
    default_model = "claude-haiku-4-5"  # Best tone for rep-facing suggestions

    async def analyze(self, text: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Generate suggestions based on all agent results.

        Args:
            text: Original customer speech.
            context: Contains all agent results:
                - sentiment, confidence, signals (from SentimentAgent)
                - persona (from PersonaAgent)
                - product (from ProductAgent)
                - competition (from CompetitionAgent)
                - sales_prompts (from SalesPromptsAgent)
                - conversation_context (history)

        Returns:
            AgentResult with suggestions list.
        """
        start = time.perf_counter()
        context = context or {}

        # Extract role and all agent data
        role = context.get("role", "Customer")
        sentiment = context.get("sentiment", "neutral")
        confidence = context.get("confidence", 0.0)
        signals = context.get("signals", [])
        conversation_context = context.get("conversation_context", "")

        persona = context.get("persona", {})
        product = context.get("product", {})
        competition = context.get("competition", {})
        sales_prompts = context.get("sales_prompts", {})

        prompts = self.prompts.get("consolidator", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "")

        # Format all agent outputs for the prompt
        user_prompt = self._format_prompt(
            user_template,
            text=text,
            role=role.upper(),
            # Sentiment
            sentiment=sentiment,
            confidence=f"{confidence:.0%}",
            signals=", ".join(signals) if signals else "none",
            # Persona
            persona_name=persona.get("persona_name", "unknown"),
            persona_segment=persona.get("segment", "unknown"),
            persona_triggers=", ".join(persona.get("detected_triggers", [])) or "none",
            # Product
            products=", ".join(product.get("products_mentioned", [])) or "none",
            ceat_products=", ".join(product.get("ceat_products_matched", [])) or "none",
            upsell_opportunities=", ".join(product.get("upsell_opportunities", [])) or "none",
            recommended_product=product.get("recommended_product", ""),
            pricing_concern="yes" if product.get("pricing_concern") else "no",
            # Competition
            competitors=", ".join(competition.get("competitors_mentioned", [])) or "none",
            counter_positioning=competition.get("counter_positioning", ""),
            competitive_concern="yes" if competition.get("competitive_concern") else "no",
            # Sales prompts
            objection=sales_prompts.get("objection_detected", "none"),
            objection_handler=sales_prompts.get("objection_handler", ""),
            upsell_type=sales_prompts.get("upsell_opportunity", "none"),
            upsell_script=sales_prompts.get("upsell_script", ""),
            call_stage=sales_prompts.get("call_stage", "discovery"),
            # Context
            conversation_context=conversation_context or "No prior context.",
        )

        response = await self._call_llm(system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - start) * 1000

        suggestions = self._parse_suggestions(response)

        return AgentResult(
            agent_name=self.name,
            success=len(suggestions) > 0,
            data={"suggestions": suggestions},
            latency_ms=latency_ms,
        )

    def _parse_suggestions(self, response: str | None) -> list[str]:
        """Parse suggestions from LLM response."""
        if not response:
            return self._fallback_suggestions()

        try:
            # Handle markdown code blocks
            content = response
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            suggestions = json.loads(content)

            if isinstance(suggestions, list):
                # Filter to strings and limit to 3
                return [str(s) for s in suggestions if s][:3]
            return self._fallback_suggestions()

        except (json.JSONDecodeError, ValueError):
            # Try to extract bullet points from plain text
            lines = response.strip().split("\n")
            suggestions = []
            for line in lines:
                line = line.strip().lstrip("-•*").strip()
                if line and len(line) < 100:
                    suggestions.append(line)
            return suggestions[:3] if suggestions else self._fallback_suggestions()

    def _fallback_suggestions(self) -> list[str]:
        """Fallback suggestions when parsing fails."""
        return [
            "Listen actively and acknowledge",
            "Ask clarifying questions",
            "Address any concerns raised",
        ]


class AgentOrchestrator:
    """Orchestrates parallel agent execution and consolidation."""

    def __init__(self, prompts: dict[str, Any]):
        self.prompts = prompts
        self.sentiment_agent = None
        self.persona_agent = None
        self.product_agent = None
        self.competition_agent = None
        self.sales_prompts_agent = None
        self.consolidator = None

    def initialize(self) -> None:
        """Initialize all agents."""
        from speechai.agents.sentiment import SentimentAgent
        from speechai.agents.persona import PersonaAgent
        from speechai.agents.product import ProductAgent
        from speechai.agents.competition import CompetitionAgent
        from speechai.agents.sales_prompts import SalesPromptsAgent

        self.sentiment_agent = SentimentAgent(self.prompts)
        self.persona_agent = PersonaAgent(self.prompts)
        self.product_agent = ProductAgent(self.prompts)
        self.competition_agent = CompetitionAgent(self.prompts)
        self.sales_prompts_agent = SalesPromptsAgent(self.prompts)
        self.consolidator = Consolidator(self.prompts)

    async def process(
        self,
        text: str,
        speaker: str = "customer",
        conversation_context: str = "",
        role: str = "Customer",
    ) -> ConsolidatedOutput:
        """Process speech through all agents in parallel.

        Args:
            text: Transcribed speech.
            speaker: Speaker identifier.
            conversation_context: Formatted conversation history for context.
            role: Speaker role for prompts ("Customer" or "Sales Rep").

        Returns:
            ConsolidatedOutput with all agent insights and suggestions.
        """
        import asyncio

        if not self.sentiment_agent or not self.consolidator:
            self.initialize()

        start = time.perf_counter()

        # Pass role to agents via context
        agent_context = {"role": role}

        # Run all parallel agents concurrently
        results = await asyncio.gather(
            self.sentiment_agent.analyze(text, context=agent_context),
            self.persona_agent.analyze(text, context=agent_context),
            self.product_agent.analyze(text, context=agent_context),
            self.competition_agent.analyze(text, context=agent_context),
            self.sales_prompts_agent.analyze(text, context=agent_context),
            return_exceptions=True,
        )

        # Collect results from parallel agents
        sentiment_result = results[0] if not isinstance(results[0], Exception) else None
        persona_result = results[1] if not isinstance(results[1], Exception) else None
        product_result = results[2] if not isinstance(results[2], Exception) else None
        competition_result = results[3] if not isinstance(results[3], Exception) else None
        sales_prompts_result = results[4] if not isinstance(results[4], Exception) else None

        # Extract data with defaults
        sentiment_data = sentiment_result.data if sentiment_result and sentiment_result.success else {"sentiment": "neutral", "confidence": 0.0, "signals": []}
        persona_data = persona_result.data if persona_result and persona_result.success else {}
        product_data = product_result.data if product_result and product_result.success else {}
        competition_data = competition_result.data if competition_result and competition_result.success else {}
        sales_prompts_data = sales_prompts_result.data if sales_prompts_result and sales_prompts_result.success else {}

        # Run consolidator with all agent outputs
        consolidator_result = await self.consolidator.analyze(
            text,
            context={
                "role": role,
                "sentiment": sentiment_data.get("sentiment", "neutral"),
                "confidence": sentiment_data.get("confidence", 0.0),
                "signals": sentiment_data.get("signals", []),
                "persona": persona_data,
                "product": product_data,
                "competition": competition_data,
                "sales_prompts": sales_prompts_data,
                "conversation_context": conversation_context,
            },
        )

        total_latency = (time.perf_counter() - start) * 1000

        suggestions = consolidator_result.data.get("suggestions", [])

        # Build enriched output with all agent data
        return ConsolidatedOutput(
            # Suggestions
            suggestions=[Suggestion(text=s) for s in suggestions],
            # Sentiment
            sentiment=sentiment_data.get("sentiment", "neutral"),
            confidence=sentiment_data.get("confidence", 0.0),
            signals=sentiment_data.get("signals", []),
            # Persona
            persona_id=persona_data.get("persona_id", ""),
            persona_name=persona_data.get("persona_name", ""),
            persona_segment=persona_data.get("segment", ""),
            # Product
            products_mentioned=product_data.get("products_mentioned", []),
            upsell_opportunities=product_data.get("upsell_opportunities", []),
            recommended_product=product_data.get("recommended_product", ""),
            # Competition
            competitors_mentioned=competition_data.get("competitors_mentioned", []),
            counter_positioning=competition_data.get("counter_positioning", ""),
            # Sales prompts
            objection_detected=sales_prompts_data.get("objection_detected", ""),
            objection_handler=sales_prompts_data.get("objection_handler", ""),
            upsell_script=sales_prompts_data.get("upsell_script", ""),
            call_stage=sales_prompts_data.get("call_stage", "discovery"),
            # Metadata
            total_latency_ms=total_latency,
        )
