"""Unified entry point for SpeechAI Sales Assistant.

Usage:
    speechai                           # mic + gemini + cli
    speechai --backend azure           # mic + azure + cli
    speechai --ui                      # mic + gemini + ui
    speechai --file recording.mp3      # file + gemini + cli
    speechai --file recording.mp3 --ui # file + gemini + ui
    speechai --file recording.mp3 --backend azure --ui
"""

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread, Timer

from dotenv import load_dotenv

from speechai.agents.consolidator import AgentOrchestrator
from speechai.context import ConversationContext
from speechai.display import Colors, DisplayConfig, clear_line, format_output, print_header, print_interim
from speechai.transcription import TranscriptResult, TranscriptSegment


def load_prompts() -> dict:
    """Load prompts from YAML file."""
    import yaml
    prompts_path = Path(__file__).parent / "prompts.yaml"
    if not prompts_path.exists():
        return {}
    with open(prompts_path) as f:
        return yaml.safe_load(f)


class SalesAssistant:
    """Unified sales assistant supporting all input/backend/interface combinations."""

    def __init__(
        self,
        source: str = "mic",  # "mic" | "file"
        backend: str = "gemini",  # "gemini" | "azure"
        interface: str = "cli",  # "cli" | "ui"
        file_path: Path | None = None,
        realtime: bool = True,
        batch_timeout_ms: int = 0,  # 0 = no batching, >0 = batch window in ms
        batch_max_count: int = 0,  # 0 = no limit, >0 = max segments before flush
    ):
        self.source = source
        self.backend = backend
        self.interface = interface
        self.file_path = file_path
        self.realtime = realtime
        self.batch_timeout_ms = batch_timeout_ms
        self.batch_max_count = batch_max_count

        # Load config and initialize components
        self.prompts = load_prompts()
        self.orchestrator = AgentOrchestrator(self.prompts)
        self.orchestrator.initialize()
        self.context = ConversationContext()

        # State
        self._running = False
        self._muted = False
        self._transcriber = None
        self._app = None  # For UI mode

        # Batching state (for Azure or when batching enabled)
        self._batch_buffer: list[TranscriptResult] = []
        self._batch_lock = Lock()
        self._batch_timer: Timer | None = None

        # Event loop for async processing (runs in background thread)
        self._loop = asyncio.new_event_loop()
        self._loop_thread: Thread | None = None

        # Display config
        self._mode_color = Colors.MAGENTA if backend == "gemini" else Colors.BLUE
        self._stt_label = f"{backend.title()} STT"
        self._output_config = self.prompts.get("output", {})

    def _get_display_config(self) -> DisplayConfig:
        """Get display configuration."""
        source_label = "File" if self.source == "file" else "Mic"
        return DisplayConfig(
            mode_name=f"Sales Assistant - {self.backend.title()} ({source_label})",
            mode_color=self._mode_color,
            pipeline_description=f"{self._stt_label} → Parallel Agents → Consolidator",
        )

    def _start_loop(self) -> None:
        """Start event loop in background thread."""
        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = Thread(target=run_loop, daemon=True)
        self._loop_thread.start()

    def _stop_loop(self) -> None:
        """Stop the background event loop."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _create_transcriber(self):
        """Create the appropriate transcriber based on backend."""
        if self.backend == "gemini":
            from speechai.transcription_gemini import GeminiTranscriber
            return GeminiTranscriber()
        else:
            from speechai.transcription import AzureTranscriber
            phrase_list = self.prompts.get("phrase_list", [])
            return AzureTranscriber(phrase_list=phrase_list)

    def _on_transcript(self, result: TranscriptResult) -> None:
        """Handle transcription result."""
        if self._muted:
            return

        if not result.is_final:
            # Interim result
            if self.interface == "cli":
                print_interim(result.speaker_id, result.text)
            elif self._app and self._app.is_running:
                self._app.call_from_thread(
                    self._app.update_interim, result.speaker_id, result.text
                )
            return

        if not result.text:
            return

        # Show interim for CLI
        if self.interface == "cli":
            clear_line()
            print_interim(result.speaker_id, result.text)

        # Check if batching is enabled
        batching_enabled = self.batch_timeout_ms > 0 or self.batch_max_count > 0

        # If batching disabled, process immediately
        if not batching_enabled:
            self._process_result(result)
            return

        # Batching enabled - buffer the result
        with self._batch_lock:
            self._batch_buffer.append(result)

            # Cancel existing timer
            if self._batch_timer:
                self._batch_timer.cancel()

            # Check if max count reached
            if self.batch_max_count > 0 and len(self._batch_buffer) >= self.batch_max_count:
                self._flush_batch()
            elif self.batch_timeout_ms > 0:
                # Start new timer
                self._batch_timer = Timer(
                    self.batch_timeout_ms / 1000.0,
                    self._flush_batch
                )
                self._batch_timer.start()

    def _flush_batch(self) -> None:
        """Flush buffered results as a single batched result."""
        with self._batch_lock:
            if not self._batch_buffer:
                return

            if self._batch_timer:
                self._batch_timer.cancel()
                self._batch_timer = None

            results = self._batch_buffer.copy()
            self._batch_buffer.clear()

        # Single result - process directly
        if len(results) == 1:
            self._process_result(results[0])
            return

        # Multiple results - create batched result
        # Handle results that may already have segments (Gemini multi-speaker)
        segments = []
        for r in results:
            if r.segments:
                # Result already has segments - use them
                segments.extend(r.segments)
            else:
                # Single-speaker result - create segment
                segments.append(TranscriptSegment(text=r.text, speaker_id=r.speaker_id))

        # Combine customer text for analysis (from all segments)
        customer_texts = [
            seg.text for seg in segments
            if self.context.assign_role(seg.speaker_id) == "customer"
        ]
        combined_text = " ".join(customer_texts) if customer_texts else segments[-1].text

        # Use customer speaker or last speaker
        primary_speaker = "Speaker-2" if customer_texts else segments[-1].speaker_id

        # Calculate total latency
        total_latency = sum(r.latency_ms for r in results)

        batch_result = TranscriptResult(
            text=combined_text,
            is_final=True,
            speaker_id=primary_speaker,
            offset_ms=results[0].offset_ms,
            latency_ms=total_latency,
            segments=segments,
        )

        self._process_result(batch_result)

    def _process_result(self, result: TranscriptResult) -> None:
        """Process a single or batched result."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._process_and_display(result), self._loop
            )
            future.result(timeout=30)
        except Exception as e:
            if self.interface == "cli":
                print(f"{Colors.DIM}[Processing error: {e}]{Colors.RESET}")
            elif self._app and self._app.is_running:
                self._app.call_from_thread(self._app.show_error, str(e))

    async def _process_and_display(self, result: TranscriptResult) -> None:
        """Process utterance through agents and display results.

        Handles both single results and batched multi-speaker results.
        For batches: stores all segments, runs agents once on combined customer text.
        """
        if self.interface == "cli":
            clear_line()

        stt_latency_ms = result.latency_ms
        timestamp = datetime.now().strftime(
            self._output_config.get("timestamp_format", "%H:%M:%S")
        )

        # Handle batched multi-speaker results
        if result.segments:
            customer_texts = []

            # Store all segments in history and display
            for segment in result.segments:
                seg_role = self.context.assign_role(segment.speaker_id)

                if seg_role == "sales_rep":
                    # Store and display sales rep segment
                    self.context.add_utterance(
                        text=segment.text,
                        speaker=segment.speaker_id,
                        role=seg_role,
                        sentiment="neutral",
                        confidence=1.0,
                        signals=[],
                    )
                    if self.interface == "cli":
                        print(f"{Colors.DIM}[{timestamp}] Rep: {segment.text}{Colors.RESET}")
                    elif self._app and self._app.is_running:
                        self._app.call_from_thread(
                            self._app._log_message,
                            f"[dim]Rep: {segment.text}[/dim]",
                        )
                else:
                    # Collect customer text for combined analysis
                    customer_texts.append(segment.text)

            # If no customer text in batch, we're done
            if not customer_texts:
                return

            # Combined customer text for single agent analysis
            combined_text = " ".join(customer_texts)
            speaker = "Speaker-2"
            role = "customer"

        else:
            # Single-speaker result (original flow)
            combined_text = result.text
            speaker = result.speaker_id
            role = self.context.assign_role(speaker)

            # Sales rep utterances: store transcript only, skip agent analysis
            if role == "sales_rep":
                self.context.add_utterance(
                    text=combined_text,
                    speaker=speaker,
                    role=role,
                    sentiment="neutral",
                    confidence=1.0,
                    signals=[],
                )
                if self.interface == "cli":
                    print(f"{Colors.DIM}[{timestamp}] Rep: {combined_text}{Colors.RESET}")
                elif self._app and self._app.is_running:
                    self._app.call_from_thread(
                        self._app._log_message,
                        f"[dim]Rep: {combined_text}[/dim]",
                    )
                return

        # Customer utterances: full agent analysis (runs once for batches)
        context_str = self.context.format_for_prompt(max_utterances=10)
        role_label = "Customer"

        output = await self.orchestrator.process(
            text=combined_text,
            speaker=speaker,
            conversation_context=context_str,
            role=role_label,
        )

        # Add customer utterance to context (for batches, this is the combined text)
        self.context.add_utterance(
            text=combined_text,
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

        # Display results
        if self.interface == "cli":
            format_output(
                timestamp=timestamp,
                speaker=speaker,
                text=combined_text,
                sentiment=output.sentiment,
                confidence=output.confidence,
                signals=output.signals,
                suggestions=output.suggestions,
                stt_latency_ms=stt_latency_ms,
                agents_latency_ms=output.total_latency_ms,
                mode_color=self._mode_color,
                stt_label=self._stt_label,
                persona_name=output.persona_name,
                persona_segment=output.persona_segment,
                products_mentioned=output.products_mentioned,
                upsell_opportunities=output.upsell_opportunities,
                recommended_product=output.recommended_product,
                competitors_mentioned=output.competitors_mentioned,
                counter_positioning=output.counter_positioning,
                objection_detected=output.objection_detected,
                upsell_script=output.upsell_script,
            )
        elif self._app and self._app.is_running:
            suggestions = [s.text for s in output.suggestions]
            self._app.call_from_thread(
                self._app.update_analysis,
                text=combined_text,
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

    def _run_mic_mode(self) -> None:
        """Run with microphone input."""
        self._transcriber = self._create_transcriber()

        # Start transcription
        if self.backend == "gemini":
            self._transcriber.start(on_result=self._on_transcript)
        else:
            self._transcriber.start(on_transcript=self._on_transcript)

        try:
            while self._running:
                time.sleep(0.1)
        finally:
            self._transcriber.stop()

    def _run_file_mode(self) -> None:
        """Run with file input."""
        if self.backend == "gemini":
            self._run_file_gemini()
        else:
            self._run_file_azure()

    def _run_file_gemini(self) -> None:
        """Process file using Gemini (frame-by-frame streaming)."""
        from speechai.transcription_file import load_audio_frames
        from speechai.transcription_gemini import GeminiTranscriber

        transcriber = GeminiTranscriber()
        transcriber.start(on_result=self._on_transcript)

        try:
            frames, frame_duration = load_audio_frames(
                self.file_path,
                sample_rate=transcriber.SAMPLE_RATE,
                frame_size=transcriber.FRAME_SIZE,
            )

            total_duration = len(frames) * frame_duration
            print(f"{Colors.DIM}Duration: {total_duration:.1f}s | Frames: {len(frames)}{Colors.RESET}")
            print(f"{Colors.DIM}Streaming...{Colors.RESET}\n")

            for i, frame in enumerate(frames):
                if not self._running:
                    break

                frame_start = time.perf_counter()
                transcriber._audio_callback(frame, transcriber.FRAME_SIZE, None, None)

                # Progress display
                elapsed_audio = i * frame_duration
                if i % int(1.0 / frame_duration) == 0:
                    progress = elapsed_audio / total_duration * 100
                    status = "speaking" if transcriber._is_speaking else "silence"
                    print(
                        f"\r{Colors.DIM}[{elapsed_audio:.1f}s / {total_duration:.1f}s] "
                        f"{progress:.0f}% - {status}{Colors.RESET}",
                        end="",
                        flush=True,
                    )

                # Real-time pacing
                if self.realtime:
                    elapsed = time.perf_counter() - frame_start
                    wait_time = frame_duration - elapsed
                    if wait_time > 0:
                        time.sleep(wait_time)

            print(f"\r{' ' * 60}\r", end="")
            print(f"{Colors.DIM}Processing final audio...{Colors.RESET}")
            time.sleep(1.5)

        finally:
            transcriber.stop()

    def _run_file_azure(self) -> None:
        """Process file using Azure (ConversationTranscriber with diarization)."""
        from speechai.transcription_file import FileTranscriberAzure

        print(f"{Colors.DIM}Starting Azure transcription with speaker diarization...{Colors.RESET}\n")

        transcriber = FileTranscriberAzure()
        transcriber.stream(self.file_path, on_result=self._on_transcript)

        print(f"\n{Colors.DIM}Azure processing complete.{Colors.RESET}")

    def _run_cli(self) -> None:
        """Run in CLI mode."""
        print_header(self._get_display_config())
        print(f"{Colors.DIM}Commands: Ctrl+C to quit{Colors.RESET}\n")

        if self.source == "file":
            self._run_file_mode()
        else:
            self._run_mic_mode()

    def _run_ui(self) -> None:
        """Run in UI mode."""
        from speechai.ui import SpeechAIApp

        self._app = SpeechAIApp()
        self._app.set_callbacks(
            on_reset=lambda: self.context.clear(),
            on_mute=lambda: setattr(self, '_muted', not self._muted),
        )

        if self.source == "file":
            # Start file streaming in background thread
            if self.backend == "gemini":
                self._transcriber = self._create_transcriber()
                self._transcriber.start(on_result=self._on_transcript)

            def stream_file():
                # Wait for app to start
                while self._app and not self._app.is_running:
                    time.sleep(0.1)
                time.sleep(0.5)

                try:
                    if self._app and self._app.is_running:
                        self._app.call_from_thread(
                            self._app._log_message,
                            f"[dim]Streaming: {self.file_path.name} ({self.backend})[/dim]",
                        )

                    if self.backend == "gemini":
                        self._stream_file_gemini_ui()
                    else:
                        self._stream_file_azure_ui()

                    if self._app and self._app.is_running:
                        self._app.call_from_thread(
                            self._app._log_message,
                            "[green]File streaming complete.[/green]",
                        )
                except Exception as e:
                    if self._app and self._app.is_running:
                        self._app.call_from_thread(self._app.show_error, str(e))

            file_thread = Thread(target=stream_file, daemon=True)
            file_thread.start()
        else:
            # Live microphone mode
            self._transcriber = self._create_transcriber()
            if self.backend == "gemini":
                self._transcriber.start(on_result=self._on_transcript)
            else:
                self._transcriber.start(on_transcript=self._on_transcript)

        try:
            self._app.run()
        finally:
            if self._transcriber:
                self._transcriber.stop()

    def _stream_file_gemini_ui(self) -> None:
        """Stream file through Gemini in UI mode."""
        from speechai.transcription_file import load_audio_frames

        frames, frame_duration = load_audio_frames(
            self.file_path,
            sample_rate=self._transcriber.SAMPLE_RATE,
            frame_size=self._transcriber.FRAME_SIZE,
        )

        for frame in frames:
            if not self._app or not self._app.is_running:
                break

            frame_start = time.perf_counter()
            self._transcriber._audio_callback(
                frame, self._transcriber.FRAME_SIZE, None, None
            )

            elapsed = time.perf_counter() - frame_start
            wait_time = frame_duration - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        time.sleep(1.5)

    def _stream_file_azure_ui(self) -> None:
        """Stream file through Azure in UI mode."""
        from speechai.transcription_file import FileTranscriberAzure

        if self._app and self._app.is_running:
            self._app.call_from_thread(
                self._app._log_message,
                "[dim]Starting Azure recognition...[/dim]",
            )

        transcriber = FileTranscriberAzure()
        transcriber.stream(self.file_path, on_result=self._on_transcript)

    def run(self) -> None:
        """Run the assistant."""
        load_dotenv()
        self._running = True
        self.context.clear()

        # Start background event loop
        self._start_loop()

        try:
            if self.interface == "ui":
                self._run_ui()
            else:
                self._run_cli()
        finally:
            self._running = False
            self._stop_loop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SpeechAI Sales Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  speechai                              # Mic + Gemini + CLI
  speechai --backend azure              # Mic + Azure + CLI
  speechai --ui                         # Mic + Gemini + UI
  speechai --file call.mp3              # File + Gemini + CLI
  speechai --file call.mp3 --ui         # File + Gemini + UI
  speechai --file call.mp3 --backend azure --ui
  speechai -b azure --batch-timeout 2000          # Batch Azure results (2s window)
  speechai -b azure --batch-max 3                 # Batch every 3 segments
        """,
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        help="Audio file to process (default: use microphone)",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["gemini", "azure"],
        default="gemini",
        help="Transcription backend (default: gemini)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Use Textual UI instead of CLI",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Process file as fast as possible (no real-time pacing)",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=0,
        metavar="MS",
        help="Batch transcripts within this time window (ms). 0=disabled (default: 0)",
    )
    parser.add_argument(
        "--batch-max",
        type=int,
        default=0,
        metavar="N",
        help="Max segments to batch before processing. 0=no limit (default: 0)",
    )
    args = parser.parse_args()

    # Validate file path
    if args.file and not args.file.exists():
        print(f"{Colors.RED}Error: File not found: {args.file}{Colors.RESET}")
        sys.exit(1)

    # Determine source
    source = "file" if args.file else "mic"
    interface = "ui" if args.ui else "cli"

    # Create and run assistant
    assistant = SalesAssistant(
        source=source,
        backend=args.backend,
        interface=interface,
        file_path=args.file,
        realtime=not args.no_realtime,
        batch_timeout_ms=args.batch_timeout,
        batch_max_count=args.batch_max,
    )

    def signal_handler(sig, frame):
        assistant._running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        assistant.run()
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
