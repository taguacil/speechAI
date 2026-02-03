"""Base agent class for parallel analysis."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import litellm


@dataclass
class AgentResult:
    """Base result from any agent."""

    agent_name: str
    success: bool
    data: dict[str, Any]
    latency_ms: float


class BaseAgent(ABC):
    """Base class for all analysis agents.

    Agents run in parallel and must be fast. Design principles:
    - Minimal prompt tokens
    - Structured JSON output
    - No retries (fail fast)
    - Async execution
    """

    name: str = "base"

    def __init__(
        self,
        prompts: dict[str, Any],
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.prompts = prompts
        self.base_url = base_url or os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
        self.model = model or os.getenv("LITELLM_MODEL", "azure/gpt-4o-mini")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "sk-1234")

    @abstractmethod
    async def analyze(self, text: str, context: dict[str, Any] | None = None) -> AgentResult:
        """Analyze text and return result.

        Args:
            text: The text to analyze (usually customer speech).
            context: Optional context (conversation history, etc.).

        Returns:
            AgentResult with analysis data.
        """
        pass

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str | None:
        """Make LLM call with minimal overhead.

        Returns raw content string or None on failure.
        """
        try:
            response = await litellm.acompletion(
                model=self.model,
                api_base=self.base_url,
                api_key=self.api_key,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=100,  # Keep responses short for speed
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[{self.name}] LLM error: {e}")
            return None

    def _format_prompt(self, template: str, **kwargs: Any) -> str:
        """Format prompt template with variables."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            print(f"[{self.name}] Missing prompt variable: {e}")
            return template
