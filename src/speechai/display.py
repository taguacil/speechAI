"""Shared display utilities for terminal output."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"

    @classmethod
    def from_name(cls, name: str) -> str:
        """Get color code from name."""
        color_map = {
            "green": cls.GREEN,
            "red": cls.RED,
            "yellow": cls.YELLOW,
            "blue": cls.BLUE,
            "cyan": cls.CYAN,
            "magenta": cls.MAGENTA,
        }
        return color_map.get(name.lower(), cls.YELLOW)


def _load_output_config() -> dict:
    """Load output config from prompts.yaml."""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    if not prompts_path.exists():
        return {}
    with open(prompts_path) as f:
        prompts = yaml.safe_load(f) or {}
    return prompts.get("output", {})


_OUTPUT_CONFIG = _load_output_config()


def _build_sentiment_colors() -> dict[str, str]:
    """Build sentiment color mapping from config."""
    config_colors = _OUTPUT_CONFIG.get("sentiment_colors", {})
    return {
        "positive": Colors.from_name(config_colors.get("positive", "green")),
        "negative": Colors.from_name(config_colors.get("negative", "red")),
        "neutral": Colors.from_name(config_colors.get("neutral", "yellow")),
    }


SENTIMENT_COLORS = _build_sentiment_colors()


@dataclass
class DisplayConfig:
    """Configuration for output display."""

    timestamp_format: str = field(
        default_factory=lambda: _OUTPUT_CONFIG.get("timestamp_format", "%H:%M:%S")
    )
    mode_name: str = field(
        default_factory=lambda: _OUTPUT_CONFIG.get("mode_name", "Sales Assistant")
    )
    mode_color: str = Colors.BLUE
    pipeline_description: str = ""


def print_header(config: DisplayConfig) -> None:
    """Print startup header."""
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  {config.mode_name}{Colors.RESET}")
    if config.pipeline_description:
        print(f"{config.mode_color}  ({config.pipeline_description}){Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.DIM}Listening... Press Ctrl+C to stop{Colors.RESET}")
    print(f"{'=' * 60}\n")


def print_interim(speaker: str, text: str) -> None:
    """Print interim transcription result."""
    truncated = text[:50] + "..." if len(text) > 50 else text
    print(f"\r{Colors.DIM}[{speaker}] {truncated}{Colors.RESET}", end="", flush=True)


def clear_line() -> None:
    """Clear the current terminal line."""
    print(f"\r{' ' * 80}\r", end="")


def format_output(
    timestamp: str,
    speaker: str,
    text: str,
    sentiment: str,
    confidence: float,
    signals: list[str],
    suggestions: list[Any],
    stt_latency_ms: float,
    agents_latency_ms: float,
    mode_color: str = Colors.BLUE,
    stt_label: str = "STT",
    # New multi-agent outputs (optional for backward compatibility)
    persona_name: str = "",
    persona_segment: str = "",
    products_mentioned: list[str] | None = None,
    upsell_opportunities: list[str] | None = None,
    recommended_product: str = "",
    competitors_mentioned: list[str] | None = None,
    counter_positioning: str = "",
    objection_detected: str = "",
    upsell_script: str = "",
    agent_latencies: dict[str, float] | None = None,
) -> None:
    """Format and print the analysis output.

    Args:
        timestamp: Formatted timestamp string.
        speaker: Speaker identifier.
        text: Original transcribed text.
        sentiment: Sentiment classification.
        confidence: Confidence score (0-1).
        signals: List of detected signals.
        suggestions: List of suggestion objects with .text attribute.
        stt_latency_ms: Speech-to-text latency.
        agents_latency_ms: Agent processing latency.
        mode_color: Color for latency display.
        stt_label: Label for STT (e.g., "Azure STT", "Gemini STT").
        persona_name: Detected customer persona name.
        persona_segment: Persona segment description.
        products_mentioned: List of products customer mentioned.
        upsell_opportunities: List of upsell opportunity keys.
        recommended_product: Recommended CEAT product.
        competitors_mentioned: List of competitors mentioned.
        counter_positioning: Counter-positioning script for competitor.
        objection_detected: Detected objection type.
        upsell_script: Upsell script to use.
    """
    sentiment_color = SENTIMENT_COLORS.get(sentiment, Colors.YELLOW)

    # Header with speaker, sentiment, and persona badge
    header_parts = [
        f"{Colors.DIM}[{timestamp}]{Colors.RESET}",
        f"{Colors.CYAN}{speaker}{Colors.RESET} │",
        f"{sentiment_color}{Colors.BOLD}{sentiment.upper()}{Colors.RESET}",
        f"{Colors.DIM}({confidence:.0%}){Colors.RESET}",
    ]
    if persona_name:
        header_parts.append(f"│ {Colors.MAGENTA}{persona_name}{Colors.RESET}")
    print(" ".join(header_parts))

    # Original text (truncated if long)
    display_text = text[:77] + "..." if len(text) > 80 else text
    print(f"  {Colors.DIM}\"{display_text}\"{Colors.RESET}")

    # Signals
    if signals:
        signals_str = ", ".join(signals)
        print(f"  {Colors.DIM}Signals: {signals_str}{Colors.RESET}")

    # Suggestions
    if suggestions:
        suggestions_config = _OUTPUT_CONFIG.get("suggestions", {})
        header = suggestions_config.get("header", "Suggestions:")
        bullet = suggestions_config.get("bullet", "→")
        bullet_color = Colors.from_name(suggestions_config.get("bullet_color", "green"))
        print(f"  {Colors.BOLD}{header}{Colors.RESET}")
        for suggestion in suggestions:
            print(f"    {bullet_color}{bullet}{Colors.RESET} {suggestion.text}")

    # Multi-agent insights (concise, one line each)
    insights = []

    # Persona insight
    if persona_segment:
        insights.append(f"Persona: {persona_name} ({persona_segment})")

    # Product insights
    if recommended_product:
        insights.append(f"Recommend: {recommended_product}")
    elif products_mentioned:
        insights.append(f"Products: {', '.join(products_mentioned[:2])}")

    # Competitor alert
    if competitors_mentioned:
        comp_list = ", ".join(competitors_mentioned[:2])
        insights.append(f"{Colors.YELLOW}Competitor: {comp_list}{Colors.RESET}")

    # Upsell opportunity
    if upsell_opportunities:
        insights.append(f"{Colors.GREEN}Upsell: {upsell_opportunities[0]}{Colors.RESET}")

    # Objection detected
    if objection_detected:
        insights.append(f"{Colors.RED}Objection: {objection_detected}{Colors.RESET}")

    # Print insights
    if insights:
        print(f"  {Colors.DIM}─{Colors.RESET}")
        for insight in insights:
            print(f"  {insight}")

    # Latency breakdown
    total_latency = stt_latency_ms + agents_latency_ms
    if agent_latencies:
        # Show individual agent latencies
        agent_parts = []
        for name, lat in agent_latencies.items():
            short_name = name[:4]  # Abbreviate: sentiment->sent, persona->pers, etc.
            agent_parts.append(f"{short_name}:{lat:.0f}")
        agents_detail = " ".join(agent_parts)
        print(
            f"  {mode_color}[{stt_label}: {stt_latency_ms:.0f}ms | "
            f"{agents_detail} | "
            f"Total: {total_latency:.0f}ms]{Colors.RESET}"
        )
    else:
        print(
            f"  {mode_color}[{stt_label}: {stt_latency_ms:.0f}ms | "
            f"Agents: {agents_latency_ms:.0f}ms | "
            f"Total: {total_latency:.0f}ms]{Colors.RESET}"
        )
    print()
