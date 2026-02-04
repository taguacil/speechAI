"""Main entry point for UI mode using Textual."""

import argparse
import asyncio
import time
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

    def __init__(self, file_path: Path | None = None, backend: str = "gemini"):
        self.prompts = load_prompts()
        self.orchestrator = AgentOrchestrator(self.prompts)
        self.orchestrator.initialize()
        self.context = ConversationContext()
        self.file_path = file_path
        self.backend = backend

        # Initialize transcriber based on backend
        if backend == "gemini":
            self.transcriber = GeminiTranscriber()
        else:
            # Azure transcriber for live mic
            from speechai.transcription import AzureTranscriber
            self.transcriber = AzureTranscriber()

        self._app: SpeechAIApp | None = None
        self._muted = False
        self._processing_loop: asyncio.AbstractEventLoop | None = None
        self._processing_thread: Thread | None = None
        self._file_thread: Thread | None = None

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
        """Handle transcript results."""
        if self._muted:
            return

        if not result.is_final:
            # Update interim display
            if self._app and self._app.is_running:
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
            # Assign role based on speaker order and call type
            role = self.context.assign_role(speaker)

            # Sales rep utterances: store transcript only, skip agent analysis
            if role == "sales_rep":
                self.context.add_utterance(
                    text=text,
                    speaker=speaker,
                    role=role,
                    sentiment="neutral",
                    confidence=1.0,
                    signals=[],
                )
                # Update UI with just the transcript (no analysis)
                if self._app and self._app.is_running:
                    self._app.call_from_thread(
                        self._app._log_message,
                        f"[dim]Rep: {text}[/dim]",
                    )
                return

            # Customer utterances: full agent analysis
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
                role=role,
                sentiment=output.sentiment,
                confidence=output.confidence,
                signals=output.signals,
                persona_name=output.persona_name,
                products_mentioned=output.products_mentioned,
                competitors_mentioned=output.competitors_mentioned,
                upsell_opportunities=output.upsell_opportunities,
            )

            # Update UI
            if self._app and self._app.is_running:
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
            if self._app and self._app.is_running:
                self._app.call_from_thread(self._app.show_error, str(e))

    def _on_reset(self) -> None:
        """Handle session reset."""
        self.context.clear()

    def _on_mute(self) -> None:
        """Handle mute toggle."""
        self._muted = not self._muted

    def _stream_file(self) -> None:
        """Stream audio file through transcriber in background."""
        if not self.file_path:
            return

        # Wait for app to be running
        while self._app and not self._app.is_running:
            time.sleep(0.1)

        # Small delay to let UI fully initialize
        time.sleep(0.5)

        try:
            if self._app and self._app.is_running:
                self._app.call_from_thread(
                    self._app._log_message,
                    f"[dim]Streaming file: {self.file_path.name} ({self.backend})[/dim]",
                )

            if self.backend == "gemini":
                self._stream_file_gemini()
            else:
                self._stream_file_azure()

            if self._app and self._app.is_running:
                self._app.call_from_thread(
                    self._app._log_message,
                    "[green]File streaming complete.[/green]",
                )

        except Exception as e:
            if self._app and self._app.is_running:
                self._app.call_from_thread(self._app.show_error, str(e))

    def _stream_file_gemini(self) -> None:
        """Stream file through Gemini transcriber."""
        from speechai.transcription_file import load_audio_frames

        frames, frame_duration = load_audio_frames(
            self.file_path,
            sample_rate=self.transcriber.SAMPLE_RATE,
            frame_size=self.transcriber.FRAME_SIZE,
        )

        for frame in frames:
            if not self._app or not self._app.is_running:
                break

            frame_start = time.perf_counter()
            self.transcriber._audio_callback(
                frame, self.transcriber.FRAME_SIZE, None, None
            )

            elapsed = time.perf_counter() - frame_start
            wait_time = frame_duration - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        time.sleep(1.5)

    def _stream_file_azure(self) -> None:
        """Stream file through Azure transcriber."""
        from speechai.transcription_file import FileTranscriberAzure

        if self._app and self._app.is_running:
            self._app.call_from_thread(
                self._app._log_message,
                "[dim]Initializing Azure transcriber...[/dim]",
            )

        try:
            transcriber = FileTranscriberAzure()
        except Exception as e:
            if self._app and self._app.is_running:
                self._app.call_from_thread(
                    self._app.show_error,
                    f"Azure init failed: {e}",
                )
            return

        if self._app and self._app.is_running:
            self._app.call_from_thread(
                self._app._log_message,
                "[dim]Starting Azure recognition...[/dim]",
            )

        def on_azure_result(result: TranscriptResult) -> None:
            """Wrapper to catch callback errors."""
            try:
                preview = result.text[:50] if len(result.text) > 50 else result.text
                if self._app and self._app.is_running:
                    self._app.call_from_thread(
                        self._app._log_message,
                        f"[cyan]Azure: {preview}[/cyan]",
                    )
                self._on_transcript(result)
            except Exception as e:
                import traceback
                if self._app and self._app.is_running:
                    self._app.call_from_thread(
                        self._app.show_error,
                        f"Callback error: {e}\n{traceback.format_exc()}",
                    )

        transcriber.stream(self.file_path, on_result=on_azure_result)

    def run(self) -> None:
        """Run the UI application."""
        load_dotenv()

        # Create and configure app
        self._app = SpeechAIApp()
        self._app.set_callbacks(on_reset=self._on_reset, on_mute=self._on_mute)

        # Start background processing
        self._start_processing_loop()

        # Start transcription based on mode
        if self.file_path:
            # File mode: stream file in background thread
            # For Azure file mode, we use FileTranscriberAzure directly
            # For Gemini file mode, we need the transcriber running for callbacks
            if self.backend == "gemini":
                self.transcriber.start(on_result=self._on_transcript)
            self._file_thread = Thread(target=self._stream_file, daemon=True)
            self._file_thread.start()
        else:
            # Live microphone mode
            if self.backend == "gemini":
                self.transcriber.start(on_result=self._on_transcript)
            else:
                self.transcriber.start(on_transcript=self._on_transcript)

        try:
            # Run Textual app (blocks until quit)
            self._app.run()
        finally:
            # Cleanup
            if not self.file_path or self.backend == "gemini":
                self.transcriber.stop()
            self._stop_processing_loop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sales Assistant with Textual UI."
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Audio file to stream (default: use microphone)",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "azure"],
        default="gemini",
        help="Transcription backend (default: gemini)",
    )
    args = parser.parse_args()

    if args.file and not args.file.exists():
        print(f"File not found: {args.file}")
        return

    assistant = UISalesAssistant(file_path=args.file, backend=args.backend)
    assistant.run()


if __name__ == "__main__":
    main()
