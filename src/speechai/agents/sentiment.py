"""Sentiment analysis agent."""

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from speechai.agents.base import AgentResult, BaseAgent


class Sentiment(Enum):
    """Sentiment classification."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Structured sentiment result."""

    sentiment: Sentiment
    confidence: float
    signals: list[str]


class SentimentAgent(BaseAgent):
    """Agent for real-time sentiment analysis of customer speech."""

    name = "sentiment"
    default_model = "gpt-5-mini"  # Reliable JSON output

    async def analyze(self, text: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Analyze sentiment of customer speech.

        Args:
            text: Customer's speech text.
            context: Optional conversation context (unused for now).

        Returns:
            AgentResult containing SentimentResult data.
        """
        start = time.perf_counter()

        if not text.strip():
            return AgentResult(
                agent_name=self.name,
                success=True,
                data={"sentiment": "neutral", "confidence": 0.0, "signals": []},
                latency_ms=0,
            )

        prompts = self.prompts.get("sentiment", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "")

        user_prompt = self._format_prompt(user_template, text=text)
        response = await self._call_llm(system_prompt, user_prompt)

        latency_ms = (time.perf_counter() - start) * 1000
        result = self._parse_response(response)

        return AgentResult(
            agent_name=self.name,
            success=result is not None,
            data=result or {"sentiment": "neutral", "confidence": 0.0, "signals": []},
            latency_ms=latency_ms,
        )

    def _parse_response(self, response: str | None) -> dict[str, Any] | None:
        """Parse LLM JSON response."""
        if not response:
            return None

        try:
            # Handle potential markdown code blocks
            content = response
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            # Validate and normalize
            sentiment = data.get("sentiment", "neutral").lower()
            if sentiment not in ("positive", "negative", "neutral"):
                sentiment = "neutral"

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            signals = data.get("signals", [])
            if not isinstance(signals, list):
                signals = [str(signals)]

            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "signals": signals[:3],  # Max 3 signals
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[sentiment] Parse error: {e}")
            return None

    @staticmethod
    def to_result(data: dict[str, Any]) -> SentimentResult:
        """Convert data dict to typed SentimentResult."""
        return SentimentResult(
            sentiment=Sentiment(data.get("sentiment", "neutral")),
            confidence=data.get("confidence", 0.0),
            signals=data.get("signals", []),
        )
