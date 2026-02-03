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
    """
    sentiment_color = SENTIMENT_COLORS.get(sentiment, Colors.YELLOW)

    # Header with speaker and sentiment
    print(
        f"{Colors.DIM}[{timestamp}]{Colors.RESET} "
        f"{Colors.CYAN}{speaker}{Colors.RESET} │ "
        f"{sentiment_color}{Colors.BOLD}{sentiment.upper()}{Colors.RESET} "
        f"{Colors.DIM}({confidence:.0%}){Colors.RESET}"
    )

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

    # Latency breakdown
    total_latency = stt_latency_ms + agents_latency_ms
    print(
        f"  {mode_color}[{stt_label}: {stt_latency_ms:.0f}ms | "
        f"Agents: {agents_latency_ms:.0f}ms | "
        f"Total: {total_latency:.0f}ms]{Colors.RESET}"
    )
    print()
