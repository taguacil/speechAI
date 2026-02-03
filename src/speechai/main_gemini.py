"""Alternative entry point using Gemini for transcription.

Uses Gemini 2.0 Flash for transcription, then the same analysis pipeline
(parallel agents + consolidator) as Azure mode for fair comparison.

Run with: uv run speechai-gemini
"""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from speechai.agents.consolidator import AgentOrchestrator, ConsolidatedOutput
from speechai.transcription_gemini import GeminiTranscriber, GeminiTranscriptResult


# ANSI colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"


SENTIMENT_COLORS = {
    "positive": Colors.GREEN,
    "negative": Colors.RED,
    "neutral": Colors.YELLOW,
}


def load_prompts() -> dict:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    if not prompts_path.exists():
        return {}
    with open(prompts_path) as f:
        return yaml.safe_load(f)


class GeminiSalesAssistant:
    """Sales assistant using Gemini transcription + standard analysis pipeline."""

    def __init__(self):
        self.prompts = load_prompts()
        self.transcriber = GeminiTranscriber()
        self.orchestrator = AgentOrchestrator(self.prompts)
        self.orchestrator.initialize()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._output_config = self.prompts.get("output", {})

    def start(self) -> None:
        """Start the assistant."""
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._print_header()

        # Start transcription with callback
        self.transcriber.start(on_result=self._on_transcript)

        # Run event loop
        try:
            while self._running:
                self._loop.run_until_complete(asyncio.sleep(0.1))
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the assistant."""
        self._running = False
        self.transcriber.stop()
        print(f"\n{Colors.DIM}Session ended.{Colors.RESET}")

    def _print_header(self) -> None:
        """Print startup header."""
        print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}  Sales Assistant - Gemini Transcription Mode{Colors.RESET}")
        print(f"{Colors.MAGENTA}  (Gemini STT → Parallel Agents → Consolidator){Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.DIM}Listening... (speak, pause, wait for analysis){Colors.RESET}")
        print(f"{Colors.DIM}Press Ctrl+C to stop{Colors.RESET}")
        print(f"{'=' * 60}\n")

    def _on_transcript(self, result: GeminiTranscriptResult) -> None:
        """Handle Gemini transcription result - pass to orchestrator."""
        if result.is_final and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._process_utterance(result),
                self._loop,
            )

    async def _process_utterance(self, transcript: GeminiTranscriptResult) -> None:
        """Process transcription through parallel agents + consolidator."""
        # Run through same orchestrator as Azure mode
        output = await self.orchestrator.process(
            text=transcript.text,
            speaker=transcript.speaker_id,
        )

        # Display results
        self._display_output(transcript, output)

    def _display_output(
        self, transcript: GeminiTranscriptResult, output: ConsolidatedOutput
    ) -> None:
        """Display analysis results to sales rep."""
        timestamp = datetime.now().strftime(
            self._output_config.get("timestamp_format", "%H:%M:%S")
        )

        sentiment_color = SENTIMENT_COLORS.get(output.sentiment, Colors.YELLOW)
        speaker = transcript.speaker_id

        # Header with speaker and sentiment
        print(
            f"{Colors.DIM}[{timestamp}]{Colors.RESET} "
            f"{Colors.CYAN}{speaker}{Colors.RESET} │ "
            f"{sentiment_color}{Colors.BOLD}{output.sentiment.upper()}{Colors.RESET} "
            f"{Colors.DIM}({output.confidence:.0%}){Colors.RESET}"
        )

        # Original text
        text = transcript.text
        if len(text) > 80:
            text = text[:77] + "..."
        print(f"  {Colors.DIM}\"{text}\"{Colors.RESET}")

        # Signals
        if output.signals:
            signals_str = ", ".join(output.signals)
            print(f"  {Colors.DIM}Signals: {signals_str}{Colors.RESET}")

        # Suggestions
        if output.suggestions:
            print(f"  {Colors.BOLD}Suggestions:{Colors.RESET}")
            for suggestion in output.suggestions:
                print(f"    {Colors.GREEN}→{Colors.RESET} {suggestion.text}")

        # Latency breakdown
        total_latency = transcript.latency_ms + output.total_latency_ms
        print(
            f"  {Colors.MAGENTA}[Gemini STT: {transcript.latency_ms:.0f}ms | "
            f"Agents: {output.total_latency_ms:.0f}ms | "
            f"Total: {total_latency:.0f}ms]{Colors.RESET}"
        )
        print()


def main() -> None:
    """Main entry point for Gemini mode."""
    load_dotenv()

    assistant = GeminiSalesAssistant()

    def signal_handler(sig, frame):
        assistant.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        assistant.start()
    except ValueError as e:
        print(f"{Colors.RED}Configuration error: {e}{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
