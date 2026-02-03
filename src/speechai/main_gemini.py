"""Alternative entry point using Gemini for transcription + analysis.

This combines transcription and sentiment analysis in a single Gemini call,
potentially reducing latency compared to Azure Speech + separate LLM.

Run with: uv run python -m speechai.main_gemini
"""

import signal
import sys
from datetime import datetime

from dotenv import load_dotenv

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


class GeminiSalesAssistant:
    """Sales assistant using Gemini for combined transcription + analysis."""

    def __init__(self):
        self.transcriber = GeminiTranscriber()
        self._running = False

    def start(self) -> None:
        """Start the assistant."""
        self._running = True
        self._print_header()

        # Start transcription with callback
        self.transcriber.start(on_result=self._on_result)

        # Keep running
        try:
            while self._running:
                import time
                time.sleep(0.1)
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
        print(f"{Colors.BOLD}  Sales Assistant - Gemini Mode{Colors.RESET}")
        print(f"{Colors.MAGENTA}  (Combined transcription + analysis){Colors.RESET}")
        print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.DIM}Listening... (speak, pause, wait for analysis){Colors.RESET}")
        print(f"{Colors.DIM}Press Ctrl+C to stop{Colors.RESET}")
        print(f"{'=' * 60}\n")

    def _on_result(self, result: GeminiTranscriptResult) -> None:
        """Handle Gemini transcription + analysis result."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        sentiment_color = SENTIMENT_COLORS.get(result.sentiment, Colors.YELLOW)

        # Header with sentiment
        print(
            f"{Colors.DIM}[{timestamp}]{Colors.RESET} "
            f"{Colors.CYAN}{result.speaker_id}{Colors.RESET} │ "
            f"{sentiment_color}{Colors.BOLD}{result.sentiment.upper()}{Colors.RESET} "
            f"{Colors.DIM}({result.confidence:.0%}){Colors.RESET}"
        )

        # Transcription
        text = result.text
        if len(text) > 80:
            text = text[:77] + "..."
        print(f"  {Colors.DIM}\"{text}\"{Colors.RESET}")

        # Signals
        if result.signals:
            signals_str = ", ".join(result.signals)
            print(f"  {Colors.DIM}Signals: {signals_str}{Colors.RESET}")

        # Latency
        print(f"  {Colors.MAGENTA}[Gemini: {result.latency_ms:.0f}ms]{Colors.RESET}")
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
