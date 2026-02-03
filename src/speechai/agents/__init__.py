"""Parallel analysis agents for sales conversations."""

from speechai.agents.base import AgentResult, BaseAgent
from speechai.agents.consolidator import Consolidator, Suggestion
from speechai.agents.sentiment import SentimentAgent, SentimentResult

__all__ = [
    "AgentResult",
    "BaseAgent",
    "Consolidator",
    "SentimentAgent",
    "SentimentResult",
    "Suggestion",
]
