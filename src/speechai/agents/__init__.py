"""Parallel analysis agents for sales conversations."""

from speechai.agents.base import AgentResult, BaseAgent
from speechai.agents.consolidator import AgentOrchestrator, Consolidator, ConsolidatedOutput, Suggestion
from speechai.agents.sentiment import SentimentAgent, SentimentResult
from speechai.agents.persona import PersonaAgent, PersonaResult, PERSONAS, PERSONA_SEGMENTS
from speechai.agents.product import ProductAgent, ProductResult, CEAT_PRODUCTS, UPSELL_TOOLKIT
from speechai.agents.competition import CompetitionAgent, CompetitionResult, COMPETITORS, CEAT_DIFFERENTIATORS, PRICE_POSITIONING
from speechai.agents.sales_prompts import SalesPromptsAgent, SalesPromptsResult, UPSELL_SCRIPTS, OBJECTION_HANDLERS, CALL_GUIDELINES, QUICK_RESPONSES

__all__ = [
    "AgentResult",
    "BaseAgent",
    "AgentOrchestrator",
    "Consolidator",
    "ConsolidatedOutput",
    "Suggestion",
    "SentimentAgent",
    "SentimentResult",
    "PersonaAgent",
    "PersonaResult",
    "PERSONAS",
    "PERSONA_SEGMENTS",
    "ProductAgent",
    "ProductResult",
    "CEAT_PRODUCTS",
    "UPSELL_TOOLKIT",
    "CompetitionAgent",
    "CompetitionResult",
    "COMPETITORS",
    "CEAT_DIFFERENTIATORS",
    "PRICE_POSITIONING",
    "SalesPromptsAgent",
    "SalesPromptsResult",
    "UPSELL_SCRIPTS",
    "OBJECTION_HANDLERS",
    "CALL_GUIDELINES",
    "QUICK_RESPONSES",
]
