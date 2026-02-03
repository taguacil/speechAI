"""Base class for sales assistant implementations."""

import asyncio
import signal
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from speechai.agents.consolidator import AgentOrchestrator, ConsolidatedOutput
from speechai.display import (
    Colors,
    DisplayConfig,
    clear_line,
    format_output,
    print_header,
    print_interim,
)


def load_prompts() -> dict:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent / "prompts.yaml"
    if not prompts_path.exists():
        return {}
    with open(prompts_path) as f:
        return yaml.safe_load(f)


class BaseSalesAssistant(ABC):
    """Base class for sales assistant implementations.

    Subclasses must implement:
    - _get_display_config(): Return display configuration
    - _start_transcription(): Start the transcription service
    - _stop_transcription(): Stop the transcription service
    """

    def __init__(self):
        self.prompts = load_prompts()
        self.orchestrator = AgentOrchestrator(self.prompts)
        self.orchestrator.initialize()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._output_config = self.prompts.get("output", {})

    @abstractmethod
    def _get_display_config(self) -> DisplayConfig:
        """Return display configuration for this mode."""
        pass

    @abstractmethod
    def _start_transcription(self) -> None:
        """Start the transcription service."""
        pass

    @abstractmethod
    def _stop_transcription(self) -> None:
        """Stop the transcription service."""
        pass

    @abstractmethod
    def _get_stt_label(self) -> str:
        """Return label for STT in latency display."""
        pass

    @abstractmethod
    def _get_mode_color(self) -> str:
        """Return color for latency display."""
        pass

    def start(self) -> None:
        """Start the sales assistant."""
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        print_header(self._get_display_config())
        self._start_transcription()

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
        self._stop_transcription()
        print(f"\n{Colors.DIM}Session ended.{Colors.RESET}")

    def _show_interim(self, speaker: str, text: str) -> None:
        """Show interim transcription."""
        print_interim(speaker, text)

    async def _process_and_display(
        self,
        text: str,
        speaker: str,
        stt_latency_ms: float,
    ) -> None:
        """Process utterance through agents and display results."""
        clear_line()

        # Run through orchestrator
        output = await self.orchestrator.process(text=text, speaker=speaker)

        # Display results
        timestamp = datetime.now().strftime(
            self._output_config.get("timestamp_format", "%H:%M:%S")
        )

        format_output(
            timestamp=timestamp,
            speaker=speaker,
            text=text,
            sentiment=output.sentiment,
            confidence=output.confidence,
            signals=output.signals,
            suggestions=output.suggestions,
            stt_latency_ms=stt_latency_ms,
            agents_latency_ms=output.total_latency_ms,
            mode_color=self._get_mode_color(),
            stt_label=self._get_stt_label(),
        )


def run_assistant(assistant: BaseSalesAssistant) -> None:
    """Run an assistant with proper signal handling."""
    load_dotenv()

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
