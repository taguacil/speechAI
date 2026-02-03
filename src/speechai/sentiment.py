"""Sentiment analysis using LiteLLM."""

import os
from dataclasses import dataclass
from enum import Enum

import litellm


class Sentiment(Enum):
    """Sentiment classification."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""

    sentiment: Sentiment
    confidence: float  # 0.0 to 1.0
    text_analyzed: str


class SentimentAnalyzer:
    """Real-time sentiment analysis using LiteLLM."""

    SYSTEM_PROMPT = """You are a sentiment classifier. Analyze the given text and respond with ONLY a JSON object in this exact format:
{"sentiment": "positive" | "negative" | "neutral", "confidence": 0.0-1.0}

Rules:
- "positive": Happy, satisfied, enthusiastic, agreeable tone
- "negative": Frustrated, angry, disappointed, objecting tone
- "neutral": Informational, factual, no strong emotion
- confidence: How certain you are (0.0 = uncertain, 1.0 = very certain)

Respond with ONLY the JSON, no other text."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.base_url = base_url or os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
        self.model = model or os.getenv("LITELLM_MODEL", "azure/gpt-4o-mini")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "sk-1234")  # LiteLLM default

    async def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text.

        Args:
            text: Text to analyze.

        Returns:
            SentimentResult with sentiment classification.
        """
        if not text.strip():
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
                text_analyzed=text,
            )

        response = await litellm.acompletion(
            model=self.model,
            api_base=self.base_url,
            api_key=self.api_key,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=50,
        )

        return self._parse_response(response, text)

    def _parse_response(self, response, text: str) -> SentimentResult:
        """Parse LLM response into SentimentResult."""
        try:
            import json

            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            sentiment = Sentiment(data.get("sentiment", "neutral").lower())
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1

            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                text_analyzed=text,
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Failed to parse sentiment response: {e}")
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                confidence=0.0,
                text_analyzed=text,
            )
