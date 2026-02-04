"""Conversation context management for session-wide tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SignalType(Enum):
    """Types of signals detected in conversation."""

    OBJECTION = "objection"
    INTEREST = "interest"
    QUESTION = "question"
    COMPETITOR = "competitor"
    BUDGET = "budget"
    TIMELINE = "timeline"
    COMMITMENT = "commitment"
    CONCERN = "concern"


@dataclass
class Utterance:
    """A single utterance in the conversation."""

    text: str
    speaker: str
    role: str  # "sales_rep" | "customer"
    timestamp: datetime
    sentiment: str
    confidence: float
    signals: list[str]
    # Multi-agent outputs
    persona_name: str = ""
    products_mentioned: list[str] = field(default_factory=list)
    competitors_mentioned: list[str] = field(default_factory=list)
    upsell_opportunities: list[str] = field(default_factory=list)


@dataclass
class TrackedSignal:
    """A structured signal extracted from conversation."""

    signal_type: SignalType
    description: str
    utterance_index: int
    timestamp: datetime


@dataclass
class ConversationContext:
    """Maintains conversation state throughout a session.

    Tracks:
    - Full utterance history
    - Structured signals (objections, interests, etc.)
    - Conversation statistics
    - Speaker roles (sales_rep vs customer)
    """

    utterances: list[Utterance] = field(default_factory=list)
    tracked_signals: list[TrackedSignal] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.now)
    call_type: str = "outbound"  # "outbound" | "inbound"
    speaker_roles: dict[str, str] = field(default_factory=dict)  # speaker_id -> role
    _last_role: str = ""  # Track last role for Unknown speaker alternation

    def assign_role(self, speaker: str) -> str:
        """Assign role to speaker based on call type and order.

        For outbound calls: first speaker = sales_rep
        For inbound calls: first speaker = customer

        Special handling for "Unknown" speakers: alternate roles based on
        conversation turn-taking assumption.
        """
        # Handle Unknown speakers with turn-based alternation
        if speaker == "Unknown":
            if not self._last_role:
                # First speaker
                role = "sales_rep" if self.call_type == "outbound" else "customer"
            else:
                # Alternate from last speaker
                role = "customer" if self._last_role == "sales_rep" else "sales_rep"
            self._last_role = role
            return role

        # Known speaker - check existing assignment
        if speaker in self.speaker_roles:
            self._last_role = self.speaker_roles[speaker]
            return self.speaker_roles[speaker]

        # First speaker assignment based on call type
        if not self.speaker_roles:
            role = "sales_rep" if self.call_type == "outbound" else "customer"
        else:
            # Subsequent speakers get the opposite role
            role = "customer" if "sales_rep" in self.speaker_roles.values() else "sales_rep"

        self.speaker_roles[speaker] = role
        self._last_role = role
        return role

    # Signal keywords for automatic extraction
    SIGNAL_KEYWORDS: dict[SignalType, list[str]] = field(default_factory=lambda: {
        SignalType.OBJECTION: [
            "too expensive", "too much", "can't afford", "over budget",
            "not sure", "hesitant", "concerned", "worried", "problem",
        ],
        SignalType.INTEREST: [
            "interested", "like that", "sounds good", "tell me more",
            "how does", "can it", "would it", "love to", "excited",
        ],
        SignalType.QUESTION: [
            "how", "what", "when", "where", "why", "can you", "could you",
            "is it possible", "do you", "does it",
        ],
        SignalType.COMPETITOR: [
            "competitor", "alternative", "other option", "also looking at",
            "compared to", "versus", "vs",
        ],
        SignalType.BUDGET: [
            "budget", "cost", "price", "pricing", "afford", "expensive",
            "cheap", "discount", "deal", "money",
        ],
        SignalType.TIMELINE: [
            "when", "timeline", "deadline", "urgent", "asap", "soon",
            "next quarter", "this month", "by end of",
        ],
        SignalType.COMMITMENT: [
            "yes", "let's do it", "sounds good", "i'm in", "sign up",
            "move forward", "next steps", "agree",
        ],
        SignalType.CONCERN: [
            "concern", "worried", "issue", "problem", "risk",
            "what if", "but", "however",
        ],
    })

    def add_utterance(
        self,
        text: str,
        speaker: str,
        role: str,
        sentiment: str,
        confidence: float,
        signals: list[str],
        # Multi-agent outputs (optional for backward compatibility)
        persona_name: str = "",
        products_mentioned: list[str] | None = None,
        competitors_mentioned: list[str] | None = None,
        upsell_opportunities: list[str] | None = None,
    ) -> None:
        """Add a new utterance and extract signals."""
        utterance = Utterance(
            text=text,
            speaker=speaker,
            role=role,
            timestamp=datetime.now(),
            sentiment=sentiment,
            confidence=confidence,
            signals=signals,
            persona_name=persona_name,
            products_mentioned=products_mentioned or [],
            competitors_mentioned=competitors_mentioned or [],
            upsell_opportunities=upsell_opportunities or [],
        )
        self.utterances.append(utterance)

        # Extract structured signals from text (only for customers)
        if role == "customer":
            self._extract_signals(utterance, len(self.utterances) - 1)

    def _extract_signals(self, utterance: Utterance, index: int) -> None:
        """Extract structured signals from utterance text."""
        text_lower = utterance.text.lower()

        for signal_type, keywords in self.SIGNAL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Avoid duplicate signals of same type from same utterance
                    existing = [
                        s for s in self.tracked_signals
                        if s.utterance_index == index and s.signal_type == signal_type
                    ]
                    if not existing:
                        self.tracked_signals.append(TrackedSignal(
                            signal_type=signal_type,
                            description=f"{keyword} detected",
                            utterance_index=index,
                            timestamp=utterance.timestamp,
                        ))
                    break  # One match per signal type per utterance

    def get_recent_utterances(self, n: int = 5) -> list[Utterance]:
        """Get the N most recent utterances."""
        return self.utterances[-n:]

    def get_signals_by_type(self, signal_type: SignalType) -> list[TrackedSignal]:
        """Get all signals of a specific type."""
        return [s for s in self.tracked_signals if s.signal_type == signal_type]

    def get_conversation_summary(self) -> dict[str, Any]:
        """Get a structured summary of the conversation."""
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        persona_counts: dict[str, int] = {}
        all_products: list[str] = []
        all_competitors: list[str] = []
        all_upsells: list[str] = []

        for u in self.utterances:
            if u.sentiment in sentiment_counts:
                sentiment_counts[u.sentiment] += 1
            if u.persona_name:
                persona_counts[u.persona_name] = persona_counts.get(u.persona_name, 0) + 1
            all_products.extend(u.products_mentioned)
            all_competitors.extend(u.competitors_mentioned)
            all_upsells.extend(u.upsell_opportunities)

        signal_counts = {}
        for s in self.tracked_signals:
            signal_counts[s.signal_type.value] = signal_counts.get(s.signal_type.value, 0) + 1

        # Deduplicate and count
        competitor_counts = {}
        for c in all_competitors:
            competitor_counts[c] = competitor_counts.get(c, 0) + 1

        return {
            "total_utterances": len(self.utterances),
            "duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "sentiment_distribution": sentiment_counts,
            "signal_counts": signal_counts,
            "has_objections": len(self.get_signals_by_type(SignalType.OBJECTION)) > 0,
            "has_interest": len(self.get_signals_by_type(SignalType.INTEREST)) > 0,
            "has_budget_discussion": len(self.get_signals_by_type(SignalType.BUDGET)) > 0,
            # Multi-agent tracking
            "persona_counts": persona_counts,
            "products_discussed": list(set(all_products)),
            "competitor_counts": competitor_counts,
            "upsell_opportunities": list(set(all_upsells)),
        }

    def format_for_prompt(self, max_utterances: int = 10) -> str:
        """Format context for inclusion in agent prompts."""
        lines = []

        # Recent conversation
        recent = self.get_recent_utterances(max_utterances)
        if recent:
            lines.append("Recent conversation:")
            for u in recent:
                sentiment_marker = {"positive": "+", "negative": "-", "neutral": "~"}
                marker = sentiment_marker.get(u.sentiment, "~")
                role_label = "Rep" if u.role == "sales_rep" else "Customer"
                lines.append(f"  [{marker}] {role_label}: \"{u.text}\"")

        # Key signals detected
        if self.tracked_signals:
            lines.append("\nKey signals detected:")
            # Group by type and show most recent
            seen_types = set()
            for s in reversed(self.tracked_signals[-10:]):
                if s.signal_type not in seen_types:
                    lines.append(f"  - {s.signal_type.value}: {s.description}")
                    seen_types.add(s.signal_type)

        # Summary stats
        summary = self.get_conversation_summary()
        if summary["total_utterances"] > 0:
            lines.append(f"\nConversation stats:")
            lines.append(f"  - {summary['total_utterances']} exchanges")
            lines.append(f"  - Sentiment: {summary['sentiment_distribution']}")
            if summary["has_objections"]:
                lines.append("  - ⚠️ Objections raised")
            if summary["has_interest"]:
                lines.append("  - ✓ Interest shown")

        return "\n".join(lines) if lines else "No conversation history yet."

    def clear(self) -> None:
        """Clear all context (for new session)."""
        self.utterances.clear()
        self.tracked_signals.clear()
        self.speaker_roles.clear()
        self._last_role = ""
        self.session_start = datetime.now()
