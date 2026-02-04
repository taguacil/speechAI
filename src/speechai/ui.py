"""Textual-based UI for real-time speech analysis display."""

import os

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static, Header, Footer, RichLog
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel


class InfoPanel(Static):
    """A panel that displays a label and value."""

    value = reactive("")

    def __init__(
        self,
        label: str,
        value: str = "-",
        panel_id: str = "",
        color: str = "white",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label = label
        self._color = color
        self.value = value
        if panel_id:
            self.id = panel_id

    def render(self) -> Text:
        text = Text()
        text.append(f"{self.label}: ", style="bold dim")
        text.append(self.value or "-", style=self._color)
        return text

    def update_value(self, value: str, color: str | None = None) -> None:
        """Update the displayed value."""
        if color:
            self._color = color
        self.value = value


class SuggestionPanel(Static):
    """Panel for displaying suggestions."""

    suggestions = reactive([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.suggestions = []

    def render(self) -> Text:
        text = Text()
        text.append("Suggestions:\n", style="bold green")
        if self.suggestions:
            for i, suggestion in enumerate(self.suggestions, 1):
                text.append(f"  {i}. ", style="dim")
                text.append(f"{suggestion}\n", style="white")
        else:
            text.append("  Waiting for input...", style="dim")
        return text

    def update_suggestions(self, suggestions: list[str]) -> None:
        """Update the suggestions list."""
        self.suggestions = list(suggestions)
        self.refresh()


class UtterancePanel(Static):
    """Panel for the current utterance."""

    text = reactive("")
    speaker = reactive("")

    def render(self) -> Text:
        result = Text()
        result.append("Last Utterance:\n", style="bold cyan")
        if self.text:
            result.append(f"  [{self.speaker}] ", style="dim")
            result.append(f'"{self.text}"', style="italic")
        else:
            result.append("  Listening...", style="dim")
        return result

    def update_utterance(self, text: str, speaker: str) -> None:
        """Update the utterance display."""
        self.speaker = speaker
        self.text = text


class SignalsPanel(Static):
    """Panel for displaying detected signals."""

    signals = reactive([])

    def render(self) -> Text:
        text = Text()
        text.append("Signals: ", style="bold dim")
        if self.signals:
            text.append(", ".join(self.signals), style="yellow")
        else:
            text.append("-", style="dim")
        return text

    def update_signals(self, signals: list[str]) -> None:
        """Update the signals list."""
        self.signals = signals


class LatencyPanel(Static):
    """Panel for displaying latency info."""

    stt_ms = reactive(0.0)
    agents_ms = reactive(0.0)
    agent_latencies: dict[str, float] = {}

    def render(self) -> Text:
        text = Text()
        total = self.stt_ms + self.agents_ms
        text.append("Latency: ", style="bold dim")
        text.append(f"STT {self.stt_ms:.0f}ms", style="blue")
        text.append(" | ", style="dim")

        if self.agent_latencies:
            # Show individual agent latencies
            for name, lat in self.agent_latencies.items():
                short = name[:4]
                text.append(f"{short}:{lat:.0f} ", style="magenta")
        else:
            text.append(f"Agents {self.agents_ms:.0f}ms", style="magenta")

        text.append("| ", style="dim")
        text.append(f"Total {total:.0f}ms", style="cyan")
        return text

    def update_latency(
        self,
        stt_ms: float,
        agents_ms: float,
        agent_latencies: dict[str, float] | None = None,
    ) -> None:
        """Update latency values."""
        self.stt_ms = stt_ms
        self.agents_ms = agents_ms
        self.agent_latencies = agent_latencies or {}
        self.refresh()


class SpeechAIApp(App):
    """Main Textual application for SpeechAI."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-container {
        height: 100%;
        padding: 1;
    }

    #top-section {
        height: auto;
        margin-bottom: 1;
    }

    #utterance-panel {
        height: auto;
        padding: 1;
        border: solid $primary;
        margin-bottom: 1;
    }

    #analysis-grid {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        height: auto;
        margin-bottom: 1;
    }

    .analysis-box {
        height: auto;
        min-height: 3;
        padding: 1;
        border: solid $secondary;
    }

    #sentiment-box {
        border: solid $success;
    }

    #persona-box {
        border: solid $warning;
    }

    #product-box {
        border: solid $primary;
    }

    #competitor-box {
        border: solid $error;
    }

    #signals-panel {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }

    #suggestions-panel {
        height: auto;
        min-height: 9;
        padding: 1;
        border: solid $success;
        margin-bottom: 0;
    }

    #latency-panel {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }

    #history-container {
        height: 1fr;
        border: solid $surface;
        padding: 1;
    }

    #history-log {
        height: 100%;
    }

    #status-bar {
        height: 1;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("r", "reset", "Reset Session"),
        ("m", "mute", "Toggle Mute"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._muted = False
        self._on_reset_callback = None
        self._on_mute_callback = None
        self._debug = os.getenv("SPEECHAI_DEBUG", "").lower() in ("1", "true", "yes")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="main-container"):
            # Current utterance
            yield UtterancePanel(id="utterance-panel")

            # Analysis grid - 2x2 layout
            with Container(id="analysis-grid"):
                with Vertical(id="sentiment-box", classes="analysis-box"):
                    yield InfoPanel("Sentiment", panel_id="sentiment-value", color="green")
                    yield InfoPanel("Confidence", panel_id="confidence-value", color="dim")

                with Vertical(id="persona-box", classes="analysis-box"):
                    yield InfoPanel("Persona", panel_id="persona-value", color="magenta")
                    yield InfoPanel("Segment", panel_id="segment-value", color="dim")

                with Vertical(id="product-box", classes="analysis-box"):
                    yield InfoPanel("Recommend", panel_id="product-value", color="cyan")
                    yield InfoPanel("Upsell", panel_id="upsell-value", color="green")

                with Vertical(id="competitor-box", classes="analysis-box"):
                    yield InfoPanel("Competitor", panel_id="competitor-value", color="red")
                    yield InfoPanel("Objection", panel_id="objection-value", color="yellow")

            # Signals
            yield SignalsPanel(id="signals-panel")

            # Suggestions
            yield SuggestionPanel(id="suggestions-panel")

            # Latency
            yield LatencyPanel(id="latency-panel")

            # History log
            with Container(id="history-container"):
                yield RichLog(id="history-log", highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.title = "SpeechAI - Real-time Sales Assistant"
        self.sub_title = "Listening..."
        self._log_message("[dim]Waiting for speech input...[/dim]")

    def _log_message(self, message: str) -> None:
        """Add message to history log."""
        log = self.query_one("#history-log", RichLog)
        log.write(message)

    def update_analysis(
        self,
        text: str,
        speaker: str,
        sentiment: str,
        confidence: float,
        signals: list[str],
        suggestions: list[str],
        stt_latency_ms: float,
        agents_latency_ms: float,
        persona_name: str = "",
        persona_segment: str = "",
        recommended_product: str = "",
        upsell_opportunities: list[str] | None = None,
        competitors_mentioned: list[str] | None = None,
        objection_detected: str = "",
        agent_latencies: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        """Update all UI panels with new analysis results."""
        if self._debug:
            self._log_message(
                f"[dim]DEBUG update_analysis: speaker={speaker}, sentiment={sentiment}, "
                f"signals={len(signals)}, suggestions={len(suggestions)}, "
                f"stt={stt_latency_ms:.0f}ms, agents={agents_latency_ms:.0f}ms[/dim]"
            )

        # Update utterance
        utterance = self.query_one("#utterance-panel", UtterancePanel)
        utterance.update_utterance(text, speaker)

        # Update sentiment with color
        sentiment_colors = {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow",
        }
        sentiment_panel = self.query_one("#sentiment-value", InfoPanel)
        sentiment_panel.update_value(
            sentiment.upper(), sentiment_colors.get(sentiment, "white")
        )

        confidence_panel = self.query_one("#confidence-value", InfoPanel)
        confidence_panel.update_value(f"{confidence:.0%}")

        # Update persona
        persona_panel = self.query_one("#persona-value", InfoPanel)
        persona_panel.update_value(persona_name or "-", "magenta")

        segment_panel = self.query_one("#segment-value", InfoPanel)
        segment_panel.update_value(persona_segment or "-")

        # Update product
        product_panel = self.query_one("#product-value", InfoPanel)
        product_panel.update_value(recommended_product or "-", "cyan")

        upsell_panel = self.query_one("#upsell-value", InfoPanel)
        upsell_text = upsell_opportunities[0] if upsell_opportunities else "-"
        upsell_panel.update_value(upsell_text, "green")

        # Update competitor
        competitor_panel = self.query_one("#competitor-value", InfoPanel)
        comp_text = ", ".join(competitors_mentioned[:2]) if competitors_mentioned else "-"
        competitor_panel.update_value(comp_text, "red" if competitors_mentioned else "dim")

        objection_panel = self.query_one("#objection-value", InfoPanel)
        objection_panel.update_value(
            objection_detected or "-", "yellow" if objection_detected else "dim"
        )

        # Update signals
        signals_panel = self.query_one("#signals-panel", SignalsPanel)
        signals_panel.update_signals(signals)

        # Update suggestions
        suggestions_panel = self.query_one("#suggestions-panel", SuggestionPanel)
        suggestions_panel.update_suggestions(suggestions)

        # Update latency
        latency_panel = self.query_one("#latency-panel", LatencyPanel)
        latency_panel.update_latency(stt_latency_ms, agents_latency_ms, agent_latencies)

        # Add to history
        sentiment_style = sentiment_colors.get(sentiment, "white")
        self._log_message(
            f"[dim]{speaker}[/dim] [{sentiment_style}]{sentiment.upper()}[/{sentiment_style}] "
            f'[italic]"{text[:60]}{"..." if len(text) > 60 else ""}"[/italic]'
        )

    def update_interim(self, speaker: str, text: str) -> None:
        """Update with interim transcription."""
        self.sub_title = f"[{speaker}] {text[:40]}..."

    def set_callbacks(self, on_reset=None, on_mute=None) -> None:
        """Set callbacks for reset and mute actions."""
        self._on_reset_callback = on_reset
        self._on_mute_callback = on_mute

    def action_reset(self) -> None:
        """Handle reset action."""
        if self._on_reset_callback:
            self._on_reset_callback()
        self._log_message("[green bold]--- Session Reset ---[/green bold]")
        # Clear panels
        self.query_one("#utterance-panel", UtterancePanel).update_utterance("", "")
        self.query_one("#suggestions-panel", SuggestionPanel).update_suggestions([])

    def action_mute(self) -> None:
        """Handle mute toggle."""
        self._muted = not self._muted
        if self._on_mute_callback:
            self._on_mute_callback()
        status = "MUTED" if self._muted else "LISTENING"
        color = "yellow" if self._muted else "green"
        self.sub_title = f"[{color}]{status}[/{color}]"
        self._log_message(f"[{color}]--- {status} ---[/{color}]")

    def show_error(self, message: str) -> None:
        """Display an error message."""
        self._log_message(f"[red bold]Error: {message}[/red bold]")
