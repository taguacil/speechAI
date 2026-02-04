"""Process audio files with real-time streaming.

Run with:
    uv run speechai-file recording.mp3
    uv run speechai-file recording.mp3 --backend azure
    uv run speechai-file recording.mp3 --no-realtime
    uv run speechai-file recordings/
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv

from speechai.agents.consolidator import AgentOrchestrator
from speechai.assistant_base import load_prompts
from speechai.context import ConversationContext
from speechai.display import Colors, clear_line, format_output, print_interim
from speechai.transcription import TranscriptResult


def get_audio_files(path: Path) -> list[Path]:
    """Get audio files from path (file or directory)."""
    audio_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    if path.is_file():
        return [path]
    elif path.is_dir():
        files = []
        for ext in audio_extensions:
            files.extend(path.glob(f"*{ext}"))
        return sorted(files)
    else:
        return []


class FileStreamProcessor:
    """Process audio files by streaming through transcriber."""

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        context: ConversationContext,
        backend: str = "gemini",
        realtime: bool = True,
    ):
        self.orchestrator = orchestrator
        self.context = context
        self.backend = backend
        self.realtime = realtime
        self._mode_color = Colors.MAGENTA if backend == "gemini" else Colors.BLUE
        self._stt_label = "Gemini STT" if backend == "gemini" else "Azure STT"

        # Event loop running in background thread for thread-safe async calls
        self._loop = asyncio.new_event_loop()
        self._loop_thread = Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

    def _run_loop(self) -> None:
        """Run the event loop in background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def close(self) -> None:
        """Clean up resources."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop and not self._loop.is_closed():
            self._loop.close()

    def process_file(self, audio_path: Path) -> None:
        """Stream audio file through transcriber and process results."""
        print(f"\n{Colors.CYAN}{'─' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}Streaming: {audio_path.name}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")

        if self.backend == "gemini":
            self._process_with_gemini(audio_path)
        else:
            self._process_with_azure(audio_path)

    def _process_with_gemini(self, audio_path: Path) -> None:
        """Process file using Gemini transcriber."""
        from speechai.transcription_file import load_audio_frames
        from speechai.transcription_gemini import GeminiTranscriber

        transcriber = GeminiTranscriber()
        transcriber.start(on_result=self._on_transcript)

        try:
            # Load audio frames
            frames, frame_duration = load_audio_frames(
                audio_path,
                sample_rate=transcriber.SAMPLE_RATE,
                frame_size=transcriber.FRAME_SIZE,
            )

            total_duration = len(frames) * frame_duration
            print(f"{Colors.DIM}Duration: {total_duration:.1f}s | Frames: {len(frames)}{Colors.RESET}")
            print(f"{Colors.DIM}Listening...{Colors.RESET}\n")

            # Feed frames to transcriber
            for i, frame in enumerate(frames):
                frame_start = time.perf_counter()

                # Feed frame to transcriber's audio callback
                transcriber._audio_callback(frame, transcriber.FRAME_SIZE, None, None)

                # Show progress every second
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

            # Clear progress line
            print(f"\r{' ' * 60}\r", end="")

            # Wait for pending transcriptions
            print(f"{Colors.DIM}Processing final audio...{Colors.RESET}")
            time.sleep(1.5)

        finally:
            transcriber.stop()

    def _process_with_azure(self, audio_path: Path) -> None:
        """Process file using Azure transcriber (streams naturally)."""
        from speechai.transcription_file import FileTranscriberAzure

        print(f"{Colors.DIM}Starting Azure continuous recognition...{Colors.RESET}\n")

        transcriber = FileTranscriberAzure()
        transcriber.stream(audio_path, on_result=self._on_transcript)

        print(f"\n{Colors.DIM}Azure processing complete.{Colors.RESET}")

    def _on_transcript(self, result: TranscriptResult) -> None:
        """Handle transcription result."""
        if not result.is_final or not result.text:
            return

        # Clear progress line and show interim
        clear_line()
        print_interim(result.speaker_id, result.text)

        # Process through agents using thread-safe async call
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._process_and_display(result), self._loop
            )
            future.result(timeout=30)  # Wait for completion
        except Exception as e:
            print(f"{Colors.DIM}[Processing error: {e}]{Colors.RESET}")

    async def _process_and_display(self, result: TranscriptResult) -> None:
        """Process utterance through agents and display."""
        clear_line()

        # Assign role based on speaker order and call type
        role = self.context.assign_role(result.speaker_id)

        # Sales rep utterances: store transcript only, skip agent analysis
        if role == "sales_rep":
            self.context.add_utterance(
                text=result.text,
                speaker=result.speaker_id,
                role=role,
                sentiment="neutral",
                confidence=1.0,
                signals=[],
            )
            # Display sales rep transcript without analysis
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{Colors.DIM}[{timestamp}] Rep: {result.text}{Colors.RESET}")
            return

        # Customer utterances: full agent analysis
        context_str = self.context.format_for_prompt(max_utterances=10)

        output = await self.orchestrator.process(
            text=result.text,
            speaker=result.speaker_id,
            conversation_context=context_str,
        )

        # Add to context
        self.context.add_utterance(
            text=result.text,
            speaker=result.speaker_id,
            role=role,
            sentiment=output.sentiment,
            confidence=output.confidence,
            signals=output.signals,
            persona_name=output.persona_name,
            products_mentioned=output.products_mentioned,
            competitors_mentioned=output.competitors_mentioned,
            upsell_opportunities=output.upsell_opportunities,
        )

        # Display
        timestamp = datetime.now().strftime("%H:%M:%S")

        format_output(
            timestamp=timestamp,
            speaker=result.speaker_id,
            text=result.text,
            sentiment=output.sentiment,
            confidence=output.confidence,
            signals=output.signals,
            suggestions=output.suggestions,
            stt_latency_ms=result.latency_ms,
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


def main() -> None:
    """Main entry point for file processing."""
    parser = argparse.ArgumentParser(
        description="Stream audio files through the sales assistant pipeline."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Audio file or directory containing audio files",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "azure"],
        default="gemini",
        help="Transcription backend (default: gemini)",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Process as fast as possible (no real-time pacing)",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"{Colors.RED}Path not found: {args.path}{Colors.RESET}")
        sys.exit(1)

    audio_files = get_audio_files(args.path)
    if not audio_files:
        print(f"{Colors.RED}No audio files found at: {args.path}{Colors.RESET}")
        sys.exit(1)

    load_dotenv()

    # Initialize orchestrator and context
    prompts = load_prompts()
    orchestrator = AgentOrchestrator(prompts)
    orchestrator.initialize()
    context = ConversationContext()

    # Print header
    backend_color = Colors.MAGENTA if args.backend == "gemini" else Colors.BLUE
    stt_name = "Gemini" if args.backend == "gemini" else "Azure"
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Sales Assistant - File Streaming Mode{Colors.RESET}")
    print(f"{backend_color}  {stt_name} STT → Parallel Agents → Consolidator{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.DIM}Files: {len(audio_files)} | Backend: {args.backend} | Realtime: {not args.no_realtime}{Colors.RESET}")

    # Process files
    processor = FileStreamProcessor(
        orchestrator=orchestrator,
        context=context,
        backend=args.backend,
        realtime=not args.no_realtime,
    )

    try:
        for audio_file in audio_files:
            processor.process_file(audio_file)
    finally:
        processor.close()

    # Print session summary
    summary = context.get_conversation_summary()
    if summary["total_utterances"] > 0:
        print(f"\n{Colors.DIM}{'─' * 40}{Colors.RESET}")
        print(f"{Colors.BOLD}Session Summary:{Colors.RESET}")
        print(f"  Files processed: {len(audio_files)}")
        print(f"  Utterances: {summary['total_utterances']}")
        print(f"  Sentiment: {summary['sentiment_distribution']}")
        if summary['signal_counts']:
            print(f"  Signals: {summary['signal_counts']}")
        if summary.get('persona_counts'):
            persona_str = ", ".join(f"{k} ({v}x)" for k, v in summary['persona_counts'].items())
            print(f"  Personas: {persona_str}")
        if summary.get('products_discussed'):
            print(f"  Products: {', '.join(summary['products_discussed'][:5])}")
        if summary.get('competitor_counts'):
            comp_str = ", ".join(f"{k} ({v}x)" for k, v in summary['competitor_counts'].items())
            print(f"  Competitors: {comp_str}")
        if summary.get('upsell_opportunities'):
            print(f"  Upsells: {', '.join(summary['upsell_opportunities'][:3])}")
        print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")

    print(f"\n{Colors.DIM}Done.{Colors.RESET}")


if __name__ == "__main__":
    main()
