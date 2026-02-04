"""Gradio-based Web UI for real-time speech analysis display."""

import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Callable

import gradio as gr


@dataclass
class AnalysisState:
    """Thread-safe shared state for analysis results."""

    # Utterance
    text: str = ""
    speaker: str = ""

    # Sentiment
    sentiment: str = ""
    confidence: float = 0.0

    # Persona
    persona_name: str = ""
    persona_segment: str = ""

    # Product
    recommended_product: str = ""
    upsell_opportunities: list[str] = field(default_factory=list)

    # Competition
    competitors_mentioned: list[str] = field(default_factory=list)
    objection_detected: str = ""

    # Signals and suggestions
    signals: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    # Latency
    stt_latency_ms: float = 0.0
    agents_latency_ms: float = 0.0

    # History
    history: list[str] = field(default_factory=list)

    # Interim text
    interim_text: str = ""
    interim_speaker: str = ""

    # Lock for thread-safe updates
    _lock: Lock = field(default_factory=Lock, repr=False)

    def update(self, **kwargs) -> None:
        """Thread-safe update of state fields."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key) and not key.startswith("_"):
                    setattr(self, key, value)

    def add_history(self, message: str) -> None:
        """Add a message to history."""
        with self._lock:
            self.history.append(message)
            # Keep last 100 messages
            if len(self.history) > 100:
                self.history = self.history[-100:]

    def get_snapshot(self) -> dict:
        """Get a thread-safe snapshot of the current state."""
        with self._lock:
            return {
                "text": self.text,
                "speaker": self.speaker,
                "sentiment": self.sentiment,
                "confidence": self.confidence,
                "persona_name": self.persona_name,
                "persona_segment": self.persona_segment,
                "recommended_product": self.recommended_product,
                "upsell_opportunities": list(self.upsell_opportunities),
                "competitors_mentioned": list(self.competitors_mentioned),
                "objection_detected": self.objection_detected,
                "signals": list(self.signals),
                "suggestions": list(self.suggestions),
                "stt_latency_ms": self.stt_latency_ms,
                "agents_latency_ms": self.agents_latency_ms,
                "history": list(self.history),
                "interim_text": self.interim_text,
                "interim_speaker": self.interim_speaker,
            }

    def clear(self) -> None:
        """Clear state (reset session)."""
        with self._lock:
            self.text = ""
            self.speaker = ""
            self.sentiment = ""
            self.confidence = 0.0
            self.persona_name = ""
            self.persona_segment = ""
            self.recommended_product = ""
            self.upsell_opportunities = []
            self.competitors_mentioned = []
            self.objection_detected = ""
            self.signals = []
            self.suggestions = []
            self.stt_latency_ms = 0.0
            self.agents_latency_ms = 0.0
            self.interim_text = ""
            self.interim_speaker = ""
            self.history.append("--- Session Reset ---")


def create_gradio_app(state: AnalysisState) -> gr.Blocks:
    """Build Gradio Blocks interface matching TUI layout."""

    def get_sentiment_color(sentiment: str) -> str:
        """Get color for sentiment badge."""
        colors = {
            "positive": "#22c55e",  # green
            "negative": "#ef4444",  # red
            "neutral": "#eab308",  # yellow
        }
        return colors.get(sentiment.lower(), "#6b7280")

    def format_utterance(text: str, speaker: str, interim_text: str, interim_speaker: str) -> str:
        """Format the utterance display."""
        if text:
            return f"**[{speaker}]** *\"{text}\"*"
        if interim_text:
            return f"**[{interim_speaker}]** *\"{interim_text}...\"* (interim)"
        return "*Listening...*"

    def format_sentiment(sentiment: str, confidence: float) -> str:
        """Format sentiment display."""
        if not sentiment:
            return "-"
        color = get_sentiment_color(sentiment)
        return f"<span style='color: {color}; font-weight: bold;'>{sentiment.upper()}</span> ({confidence:.0%})"

    def format_persona(name: str, segment: str) -> str:
        """Format persona display."""
        if not name:
            return "-"
        parts = [f"**{name}**"]
        if segment:
            parts.append(f"({segment})")
        return " ".join(parts)

    def format_product(recommended: str, upsell: list) -> str:
        """Format product display."""
        parts = []
        if recommended:
            parts.append(f"Recommend: **{recommended}**")
        if upsell:
            parts.append(f"Upsell: {upsell[0]}")
        return " | ".join(parts) if parts else "-"

    def format_competitor(competitors: list, objection: str) -> str:
        """Format competitor display."""
        parts = []
        if competitors:
            comp_text = ", ".join(competitors[:2])
            parts.append(f"<span style='color: #ef4444;'>{comp_text}</span>")
        if objection:
            parts.append(f"Objection: <span style='color: #eab308;'>{objection}</span>")
        return " | ".join(parts) if parts else "-"

    def format_signals(signals: list) -> str:
        """Format signals display."""
        if not signals:
            return "-"
        return ", ".join(f"<span style='color: #eab308;'>{s}</span>" for s in signals)

    def format_suggestions(suggestions: list) -> str:
        """Format suggestions as numbered list."""
        if not suggestions:
            return "*Waiting for input...*"
        return "\n".join(f"{i}. {s}" for i, s in enumerate(suggestions, 1))

    def format_latency(stt_ms: float, agents_ms: float) -> str:
        """Format latency display."""
        total = stt_ms + agents_ms
        return (
            f"<span style='color: #3b82f6;'>STT {stt_ms:.0f}ms</span> | "
            f"<span style='color: #a855f7;'>Agents {agents_ms:.0f}ms</span> | "
            f"<span style='color: #06b6d4;'>Total {total:.0f}ms</span>"
        )

    def format_history(history: list) -> str:
        """Format history as scrollable text."""
        if not history:
            return "*Waiting for speech input...*"
        return "\n".join(history[-20:])  # Show last 20 entries

    def poll_state():
        """Poll state and return updated values for all components."""
        snapshot = state.get_snapshot()

        return (
            format_utterance(
                snapshot["text"],
                snapshot["speaker"],
                snapshot["interim_text"],
                snapshot["interim_speaker"],
            ),
            format_sentiment(snapshot["sentiment"], snapshot["confidence"]),
            format_persona(snapshot["persona_name"], snapshot["persona_segment"]),
            format_product(snapshot["recommended_product"], snapshot["upsell_opportunities"]),
            format_competitor(snapshot["competitors_mentioned"], snapshot["objection_detected"]),
            format_signals(snapshot["signals"]),
            format_suggestions(snapshot["suggestions"]),
            format_latency(snapshot["stt_latency_ms"], snapshot["agents_latency_ms"]),
            format_history(snapshot["history"]),
        )

    # Custom CSS for styling
    custom_css = """
    .utterance-panel {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3d 100%);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
    }
    .analysis-box {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 12px;
        min-height: 60px;
    }
    .suggestions-box {
        background: linear-gradient(135deg, #1e3a1e 0%, #2d3d2d 100%);
        border-radius: 8px;
        padding: 16px;
    }
    .history-box {
        background: #0d0d0d;
        border-radius: 8px;
        padding: 12px;
        font-family: monospace;
        font-size: 12px;
        max-height: 200px;
        overflow-y: auto;
    }
    """

    with gr.Blocks(
        title="SpeechAI - Real-time Sales Assistant",
        theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="purple"),
        css=custom_css,
    ) as app:
        gr.Markdown("# SpeechAI - Real-time Sales Assistant")

        # Last Utterance
        with gr.Group():
            utterance_display = gr.Markdown(
                "*Listening...*",
                elem_classes=["utterance-panel"],
            )

        # 2x2 Grid for Analysis
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Sentiment")
                sentiment_display = gr.Markdown("-", elem_classes=["analysis-box"])
            with gr.Column():
                gr.Markdown("### Persona")
                persona_display = gr.Markdown("-", elem_classes=["analysis-box"])

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Product")
                product_display = gr.Markdown("-", elem_classes=["analysis-box"])
            with gr.Column():
                gr.Markdown("### Competition")
                competitor_display = gr.Markdown("-", elem_classes=["analysis-box"])

        # Signals
        with gr.Group():
            gr.Markdown("### Signals")
            signals_display = gr.Markdown("-")

        # Suggestions
        with gr.Group():
            gr.Markdown("### Suggestions")
            suggestions_display = gr.Markdown(
                "*Waiting for input...*",
                elem_classes=["suggestions-box"],
            )

        # Latency
        with gr.Group():
            latency_display = gr.Markdown(
                "<span style='color: #6b7280;'>Latency: -</span>"
            )

        # Conversation History
        with gr.Accordion("Conversation History", open=True):
            history_display = gr.Markdown(
                "*Waiting for speech input...*",
                elem_classes=["history-box"],
            )

        # Timer for polling
        timer = gr.Timer(0.2)  # 200ms interval
        timer.tick(
            fn=poll_state,
            outputs=[
                utterance_display,
                sentiment_display,
                persona_display,
                product_display,
                competitor_display,
                signals_display,
                suggestions_display,
                latency_display,
                history_display,
            ],
        )

    return app


class WebUIAdapter:
    """Adapter implementing same interface as SpeechAIApp for web UI."""

    def __init__(self):
        self.state = AnalysisState()
        self._app: gr.Blocks | None = None
        self._is_running = False
        self._muted = False
        self._on_reset_callback: Callable | None = None
        self._on_mute_callback: Callable | None = None
        self._debug = os.getenv("SPEECHAI_DEBUG", "").lower() in ("1", "true", "yes")

    @property
    def is_running(self) -> bool:
        """Check if the app is running."""
        return self._is_running

    def set_callbacks(self, on_reset: Callable | None = None, on_mute: Callable | None = None) -> None:
        """Set callbacks for reset and mute actions."""
        self._on_reset_callback = on_reset
        self._on_mute_callback = on_mute

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
        **kwargs,
    ) -> None:
        """Update state with new analysis results."""
        if self._debug:
            print(
                f"[DEBUG] WebUI update_analysis: speaker={speaker}, sentiment={sentiment}, "
                f"signals={len(signals)}, suggestions={len(suggestions)}, "
                f"stt={stt_latency_ms:.0f}ms, agents={agents_latency_ms:.0f}ms"
            )
        self.state.update(
            text=text,
            speaker=speaker,
            sentiment=sentiment,
            confidence=confidence,
            signals=signals,
            suggestions=suggestions,
            stt_latency_ms=stt_latency_ms,
            agents_latency_ms=agents_latency_ms,
            persona_name=persona_name,
            persona_segment=persona_segment,
            recommended_product=recommended_product,
            upsell_opportunities=upsell_opportunities or [],
            competitors_mentioned=competitors_mentioned or [],
            objection_detected=objection_detected,
            interim_text="",  # Clear interim when final arrives
            interim_speaker="",
        )

        # Add to history
        sentiment_colors = {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow",
        }
        color = sentiment_colors.get(sentiment, "white")
        truncated = text[:60] + "..." if len(text) > 60 else text
        self.state.add_history(f"[{speaker}] {sentiment.upper()} \"{truncated}\"")

    def update_interim(self, speaker: str, text: str) -> None:
        """Update with interim transcription."""
        self.state.update(interim_speaker=speaker, interim_text=text)

    def show_error(self, message: str) -> None:
        """Display an error message."""
        self.state.add_history(f"ERROR: {message}")

    def _log_message(self, message: str) -> None:
        """Add message to history log."""
        # Strip rich markup for web display
        clean_msg = message.replace("[dim]", "").replace("[/dim]", "")
        clean_msg = clean_msg.replace("[green]", "").replace("[/green]", "")
        clean_msg = clean_msg.replace("[red]", "").replace("[/red]", "")
        clean_msg = clean_msg.replace("[bold]", "").replace("[/bold]", "")
        self.state.add_history(clean_msg)

    def call_from_thread(self, func: Callable, *args, **kwargs) -> None:
        """Call a function from a background thread (safe in Gradio)."""
        # Gradio state is thread-safe, just call directly
        func(*args, **kwargs)

    def run(self) -> None:
        """Run the Gradio app."""
        self._app = create_gradio_app(self.state)
        self._is_running = True
        try:
            self._app.launch(
                server_name="127.0.0.1",
                server_port=7860,
                share=False,
                show_error=True,
                quiet=False,
            )
        finally:
            self._is_running = False
