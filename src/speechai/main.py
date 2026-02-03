"""Main entry point for real-time speech sentiment analysis."""

import asyncio
import signal
import sys
from datetime import datetime

from dotenv import load_dotenv

from speechai.sentiment import Sentiment, SentimentAnalyzer
from speechai.transcription import AzureTranscriber, TranscriptResult


# ANSI colors for terminal output
COLORS = {
    Sentiment.POSITIVE: "\033[92m",  # Green
    Sentiment.NEGATIVE: "\033[91m",  # Red
    Sentiment.NEUTRAL: "\033[93m",   # Yellow
}
RESET = "\033[0m"
BOLD = "\033[1m"


class SpeechSentimentAnalyzer:
    """Real-time speech sentiment analysis pipeline."""

    def __init__(self):
        self.transcriber = AzureTranscriber()
        self.sentiment_analyzer = SentimentAnalyzer()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

    def start(self) -> None:
        """Start the analysis pipeline."""
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        print(f"{BOLD}Speech Sentiment Analyzer{RESET}")
        print("=" * 40)
        print("Listening... (Ctrl+C to stop)\n")

        # Start transcription with callback
        self.transcriber.start(on_transcript=self._on_transcript)

        # Keep running until stopped
        try:
            while self._running:
                self._loop.run_until_complete(asyncio.sleep(0.1))
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the analysis pipeline."""
        self._running = False
        self.transcriber.stop()
        print("\n\nStopped.")

    def _on_transcript(self, result: TranscriptResult) -> None:
        """Handle transcript results - analyze sentiment on final results."""
        if result.is_final and self._loop:
            # Run sentiment analysis asynchronously
            asyncio.run_coroutine_threadsafe(
                self._analyze_and_display(result.text),
                self._loop,
            )
        elif not result.is_final:
            # Show interim results (what's being spoken)
            print(f"\r{BOLD}[listening]{RESET} {result.text[:60]}...", end="", flush=True)

    async def _analyze_and_display(self, text: str) -> None:
        """Analyze sentiment and display results."""
        result = await self.sentiment_analyzer.analyze(text)

        color = COLORS.get(result.sentiment, "")
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Clear the interim line and print final result
        print(f"\r{' ' * 80}\r", end="")  # Clear line
        print(
            f"[{timestamp}] "
            f"{color}{BOLD}{result.sentiment.value.upper():8}{RESET} "
            f"({result.confidence:.0%}) "
            f'"{text}"'
        )


def main() -> None:
    """Main entry point."""
    load_dotenv()

    analyzer = SpeechSentimentAnalyzer()

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        analyzer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        analyzer.start()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
