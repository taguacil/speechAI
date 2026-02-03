"""Main entry point for real-time sales assistant."""

import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from speechai.agents.consolidator import AgentOrchestrator, ConsolidatedOutput
from speechai.transcription import AzureTranscriber, TranscriptResult


# ANSI colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"


SENTIMENT_COLORS = {
    "positive": Colors.GREEN,
    "negative": Colors.RED,
    "neutral": Colors.YELLOW,
}


def load_prompts() -> dict:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    if not prompts_path.exists():
        print(f"Warning: prompts.yaml not found at {prompts_path}")
        return {}

    with open(prompts_path) as f:
        return yaml.safe_load(f)


class SalesAssistant:
    """Real-time sales assistant with speaker diarization and parallel analysis."""

    def __init__(self):
        self.prompts = load_prompts()

        # Get phrase list from prompts
        phrase_list = self.prompts.get("phrase_list", [])

        self.transcriber = AzureTranscriber(phrase_list=phrase_list)
        self.orchestrator = AgentOrchestrator(self.prompts)
        self.orchestrator.initialize()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

        # Track which speaker is which (can be set during call)
        self._customer_speaker: str | None = None

        # Output config
        self._output_config = self.prompts.get("output", {})

    def start(self) -> None:
        """Start the sales assistant."""
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._print_header()

        # Start transcription
        self.transcriber.start(on_transcript=self._on_transcript)

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
        print(f"{Colors.BOLD}  Sales Assistant - Real-time Analysis{Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.DIM}Listening with speaker diarization...{Colors.RESET}")
        print(f"{Colors.DIM}Press Ctrl+C to stop{Colors.RESET}")
        print(f"{'=' * 60}\n")

    def _on_transcript(self, result: TranscriptResult) -> None:
        """Handle transcript results."""
        if result.is_final and self._loop:
            # Process final utterances through agents
            asyncio.run_coroutine_threadsafe(
                self._process_utterance(result),
                self._loop,
            )
        elif not result.is_final:
            # Show interim results
            self._show_interim(result)

    def _show_interim(self, result: TranscriptResult) -> None:
        """Show interim transcription (while speaking)."""
        speaker = result.speaker_id
        text = result.text[:50] + "..." if len(result.text) > 50 else result.text
        print(f"\r{Colors.DIM}[{speaker}] {text}{Colors.RESET}", end="", flush=True)

    async def _process_utterance(self, result: TranscriptResult) -> None:
        """Process a complete utterance through all agents."""
        # Clear interim line
        print(f"\r{' ' * 80}\r", end="")

        # Run parallel agents and consolidate
        output = await self.orchestrator.process(
            text=result.text,
            speaker=result.speaker_id,
        )

        # Display results
        self._display_output(result, output)

    def _display_output(
        self, transcript: TranscriptResult, output: ConsolidatedOutput
    ) -> None:
        """Display analysis results to sales rep."""
        timestamp = datetime.now().strftime(
            self._output_config.get("timestamp_format", "%H:%M:%S")
        )

        # Sentiment color
        sentiment_color = SENTIMENT_COLORS.get(output.sentiment, Colors.YELLOW)

        # Speaker label
        speaker = transcript.speaker_id

        # Header with speaker and sentiment
        print(
            f"{Colors.DIM}[{timestamp}]{Colors.RESET} "
            f"{Colors.CYAN}{speaker}{Colors.RESET} │ "
            f"{sentiment_color}{Colors.BOLD}{output.sentiment.upper()}{Colors.RESET} "
            f"{Colors.DIM}({output.confidence:.0%}){Colors.RESET}"
        )

        # Original text (truncated if long)
        text = transcript.text
        if len(text) > 80:
            text = text[:77] + "..."
        print(f"  {Colors.DIM}\"{text}\"{Colors.RESET}")

        # Signals (if any)
        if output.signals:
            signals_str = ", ".join(output.signals)
            print(f"  {Colors.DIM}Signals: {signals_str}{Colors.RESET}")

        # Suggestions (the key output for sales rep)
        if output.suggestions:
            print(f"  {Colors.BOLD}Suggestions:{Colors.RESET}")
            for suggestion in output.suggestions:
                print(f"    {Colors.GREEN}→{Colors.RESET} {suggestion.text}")

        # Latency breakdown (same format as Gemini for comparison)
        total_latency = transcript.latency_ms + output.total_latency_ms
        print(
            f"  {Colors.BLUE}[Azure STT: {transcript.latency_ms:.0f}ms | "
            f"Agents: {output.total_latency_ms:.0f}ms | "
            f"Total: {total_latency:.0f}ms]{Colors.RESET}"
        )
        print()


def main() -> None:
    """Main entry point."""
    load_dotenv()

    assistant = SalesAssistant()

    # Graceful shutdown
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
        sys.exit(1)


if __name__ == "__main__":
    main()
