"""Consolidator that combines agent outputs into actionable suggestions."""

import json
import time
from dataclasses import dataclass
from typing import Any

from speechai.agents.base import AgentResult, BaseAgent


@dataclass
class Suggestion:
    """A single suggestion for the sales rep."""

    text: str


@dataclass
class ConsolidatedOutput:
    """Final output shown to sales rep."""

    suggestions: list[Suggestion]
    sentiment: str
    confidence: float
    signals: list[str]
    total_latency_ms: float


class Consolidator(BaseAgent):
    """Consolidates parallel agent outputs into actionable suggestions.

    This runs AFTER all parallel agents complete and combines their
    results into 2-3 bullet suggestions for the sales rep.
    """

    name = "consolidator"

    async def analyze(self, text: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Generate suggestions based on agent results.

        Args:
            text: Original customer speech.
            context: Must contain agent results:
                - sentiment: str
                - confidence: float
                - signals: list[str]

        Returns:
            AgentResult with suggestions list.
        """
        start = time.perf_counter()
        context = context or {}

        sentiment = context.get("sentiment", "neutral")
        confidence = context.get("confidence", 0.0)
        signals = context.get("signals", [])
        conversation_context = context.get("conversation_context", "")

        prompts = self.prompts.get("consolidator", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "")

        user_prompt = self._format_prompt(
            user_template,
            text=text,
            sentiment=sentiment,
            confidence=f"{confidence:.0%}",
            signals=", ".join(signals) if signals else "none detected",
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
        self.consolidator = None
        # Future agents can be added here:
        # self.sale_progress_agent = None
        # self.objection_agent = None

    def initialize(self) -> None:
        """Initialize all agents."""
        from speechai.agents.sentiment import SentimentAgent

        self.sentiment_agent = SentimentAgent(self.prompts)
        self.consolidator = Consolidator(self.prompts)

    async def process(
        self,
        text: str,
        speaker: str = "customer",
        conversation_context: str = "",
    ) -> ConsolidatedOutput:
        """Process customer speech through all agents.

        Args:
            text: Transcribed speech.
            speaker: Speaker identifier.
            conversation_context: Formatted conversation history for context.

        Returns:
            ConsolidatedOutput with suggestions and analysis.
        """
        import asyncio

        if not self.sentiment_agent or not self.consolidator:
            self.initialize()

        start = time.perf_counter()

        # Run parallel agents (currently just sentiment, more can be added)
        results = await asyncio.gather(
            self.sentiment_agent.analyze(text),
            # Add more agents here for parallel execution:
            # self.sale_progress_agent.analyze(text, context),
            return_exceptions=True,
        )

        # Collect results
        sentiment_result = results[0] if not isinstance(results[0], Exception) else None

        # Extract sentiment data
        if sentiment_result and sentiment_result.success:
            sentiment_data = sentiment_result.data
        else:
            sentiment_data = {"sentiment": "neutral", "confidence": 0.0, "signals": []}

        # Run consolidator with combined context including conversation history
        consolidator_result = await self.consolidator.analyze(
            text,
            context={
                "sentiment": sentiment_data["sentiment"],
                "confidence": sentiment_data["confidence"],
                "signals": sentiment_data["signals"],
                "conversation_context": conversation_context,
            },
        )

        total_latency = (time.perf_counter() - start) * 1000

        suggestions = consolidator_result.data.get("suggestions", [])

        return ConsolidatedOutput(
            suggestions=[Suggestion(text=s) for s in suggestions],
            sentiment=sentiment_data["sentiment"],
            confidence=sentiment_data["confidence"],
            signals=sentiment_data["signals"],
            total_latency_ms=total_latency,
        )
