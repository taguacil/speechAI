"""Main entry point for UI mode using Textual."""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path
from threading import Thread

import yaml
from dotenv import load_dotenv

from speechai.agents.consolidator import AgentOrchestrator
from speechai.context import ConversationContext
from speechai.transcription import TranscriptResult
from speechai.transcription_gemini import GeminiTranscriber
from speechai.ui import SpeechAIApp


def load_prompts() -> dict:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    if not prompts_path.exists():
        return {}
    with open(prompts_path) as f:
        return yaml.safe_load(f)


class UISalesAssistant:
    """Sales assistant with Textual UI."""

    def __init__(self):
        self.prompts = load_prompts()
        self.orchestrator = AgentOrchestrator(self.prompts)
        self.orchestrator.initialize()
        self.context = ConversationContext()
        self.transcriber = GeminiTranscriber()

        self._app: SpeechAIApp | None = None
        self._muted = False
        self._processing_loop: asyncio.AbstractEventLoop | None = None
        self._processing_thread: Thread | None = None

    def _start_processing_loop(self) -> None:
        """Start background thread for async processing."""
        self._processing_loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(self._processing_loop)
            self._processing_loop.run_forever()

        self._processing_thread = Thread(target=run_loop, daemon=True)
        self._processing_thread.start()

    def _stop_processing_loop(self) -> None:
        """Stop the background processing loop."""
        if self._processing_loop:
            self._processing_loop.call_soon_threadsafe(self._processing_loop.stop)

    def _on_transcript(self, result: TranscriptResult) -> None:
        """Handle transcript results from Gemini."""
        if self._muted:
            return

        if not result.is_final:
            # Update interim display
            if self._app:
                self._app.call_from_thread(
                    self._app.update_interim, result.speaker_id, result.text
                )
            return

        # Process final transcript in background
        if self._processing_loop:
            asyncio.run_coroutine_threadsafe(
                self._process_utterance(
                    text=result.text,
                    speaker=result.speaker_id,
                    stt_latency_ms=result.latency_ms,
                ),
                self._processing_loop,
            )

    async def _process_utterance(
        self, text: str, speaker: str, stt_latency_ms: float
    ) -> None:
        """Process utterance through agents and update UI."""
        try:
            # Get context for agents
            context_str = self.context.format_for_prompt(max_utterances=10)

            # Run through orchestrator
            output = await self.orchestrator.process(
                text=text,
                speaker=speaker,
                conversation_context=context_str,
            )

            # Add to conversation context
            self.context.add_utterance(
                text=text,
                speaker=speaker,
                sentiment=output.sentiment,
                confidence=output.confidence,
                signals=output.signals,
                persona_name=output.persona_name,
                products_mentioned=output.products_mentioned,
                competitors_mentioned=output.competitors_mentioned,
                upsell_opportunities=output.upsell_opportunities,
            )

            # Update UI
            if self._app:
                suggestions = [s.text for s in output.suggestions]
                self._app.call_from_thread(
                    self._app.update_analysis,
                    text=text,
                    speaker=speaker,
                    sentiment=output.sentiment,
                    confidence=output.confidence,
                    signals=output.signals,
                    suggestions=suggestions,
                    stt_latency_ms=stt_latency_ms,
                    agents_latency_ms=output.total_latency_ms,
                    persona_name=output.persona_name,
                    persona_segment=output.persona_segment,
                    recommended_product=output.recommended_product,
                    upsell_opportunities=output.upsell_opportunities,
                    competitors_mentioned=output.competitors_mentioned,
                    objection_detected=output.objection_detected,
                )
        except Exception as e:
            if self._app:
                self._app.call_from_thread(self._app.show_error, str(e))

    def _on_reset(self) -> None:
        """Handle session reset."""
        self.context.clear()

    def _on_mute(self) -> None:
        """Handle mute toggle."""
        self._muted = not self._muted

    def run(self) -> None:
        """Run the UI application."""
        load_dotenv()

        # Create and configure app
        self._app = SpeechAIApp()
        self._app.set_callbacks(on_reset=self._on_reset, on_mute=self._on_mute)

        # Start background processing
        self._start_processing_loop()

        # Start transcription
        self.transcriber.start(on_result=self._on_transcript)

        try:
            # Run Textual app (blocks until quit)
            self._app.run()
        finally:
            # Cleanup
            self.transcriber.stop()
            self._stop_processing_loop()


def main() -> None:
    """Main entry point."""
    assistant = UISalesAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
