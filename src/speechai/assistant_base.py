"""Base class for sales assistant implementations."""

import asyncio
import select
import signal
import sys
import termios
import tty
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from threading import Thread

import yaml
from dotenv import load_dotenv

from speechai.agents.consolidator import AgentOrchestrator, ConsolidatedOutput
from speechai.context import ConversationContext
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

    Keyboard commands during session:
    - r: Reset session (clear context, start fresh)
    - q: Quit
    """

    def __init__(self):
        self.prompts = load_prompts()
        self.orchestrator = AgentOrchestrator(self.prompts)
        self.orchestrator.initialize()
        self.context = ConversationContext()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._muted = False
        self._output_config = self.prompts.get("output", {})
        self._keyboard_thread: Thread | None = None
        self._old_terminal_settings = None

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

    def _print_session_summary(self) -> None:
        """Print summary of current session."""
        summary = self.context.get_conversation_summary()
        if summary["total_utterances"] > 0:
            print(f"\n{Colors.DIM}{'─' * 40}{Colors.RESET}")
            print(f"{Colors.BOLD}Session Summary:{Colors.RESET}")
            print(f"  Utterances: {summary['total_utterances']}")
            print(f"  Duration: {summary['duration_seconds']:.0f}s")
            print(f"  Sentiment: {summary['sentiment_distribution']}")
            if summary['signal_counts']:
                print(f"  Signals: {summary['signal_counts']}")
            # Multi-agent tracking
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

    def reset_session(self) -> None:
        """Reset the session - clear context and start fresh."""
        # Print summary of old session
        self._print_session_summary()

        # Clear context
        self.context.clear()

        # Print reset message
        print(f"\n{Colors.GREEN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}  SESSION RESET - Starting fresh{Colors.RESET}")
        print(f"{Colors.GREEN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.DIM}Listening... [r]=reset [m]=mute [q]=quit{Colors.RESET}\n")

    def toggle_mute(self) -> None:
        """Toggle mute state."""
        self._muted = not self._muted
        if self._muted:
            print(f"\n{Colors.YELLOW}{'─' * 40}{Colors.RESET}")
            print(f"{Colors.YELLOW}{Colors.BOLD}  MUTED - Press [m] to resume{Colors.RESET}")
            print(f"{Colors.YELLOW}{'─' * 40}{Colors.RESET}\n")
        else:
            print(f"\n{Colors.GREEN}{'─' * 40}{Colors.RESET}")
            print(f"{Colors.GREEN}{Colors.BOLD}  UNMUTED - Listening{Colors.RESET}")
            print(f"{Colors.GREEN}{'─' * 40}{Colors.RESET}\n")

    def _start_keyboard_listener(self) -> None:
        """Start listening for keyboard input in a separate thread."""
        if not sys.stdin.isatty():
            return  # Don't listen if not a terminal

        def keyboard_loop():
            try:
                # Save terminal settings and set to raw mode
                self._old_terminal_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())

                while self._running:
                    # Check if input is available (with timeout)
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key.lower() == 'r':
                            self.reset_session()
                        elif key.lower() == 'm':
                            self.toggle_mute()
                        elif key.lower() == 'q':
                            self._running = False
            except Exception:
                pass  # Ignore keyboard errors
            finally:
                # Restore terminal settings
                if self._old_terminal_settings:
                    try:
                        termios.tcsetattr(
                            sys.stdin, termios.TCSADRAIN, self._old_terminal_settings
                        )
                    except Exception:
                        pass

        self._keyboard_thread = Thread(target=keyboard_loop, daemon=True)
        self._keyboard_thread.start()

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener and restore terminal."""
        if self._old_terminal_settings:
            try:
                termios.tcsetattr(
                    sys.stdin, termios.TCSADRAIN, self._old_terminal_settings
                )
            except Exception:
                pass

    def start(self) -> None:
        """Start the sales assistant."""
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.context.clear()  # Fresh context for new session

        print_header(self._get_display_config())
        print(f"{Colors.DIM}Commands: [r]=reset [m]=mute [q]=quit{Colors.RESET}\n")

        self._start_transcription()
        self._start_keyboard_listener()

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
        self._stop_keyboard_listener()
        self._stop_transcription()

        # Print final session summary
        self._print_session_summary()
        print(f"{Colors.DIM}Session ended.{Colors.RESET}")

    def _show_interim(self, speaker: str, text: str) -> None:
        """Show interim transcription."""
        if not self._muted:
            print_interim(speaker, text)

    async def _process_and_display(
        self,
        text: str,
        speaker: str,
        stt_latency_ms: float,
    ) -> None:
        """Process utterance through agents and display results."""
        # Skip processing if muted
        if self._muted:
            clear_line()
            return

        clear_line()

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
            # Display sales rep transcript without analysis
            timestamp = datetime.now().strftime(
                self._output_config.get("timestamp_format", "%H:%M:%S")
            )
            print(
                f"{Colors.DIM}[{timestamp}] Rep: {text}{Colors.RESET}"
            )
            return

        # Customer utterances: full agent analysis
        # Get formatted context for agents
        context_str = self.context.format_for_prompt(max_utterances=10)

        # Run through orchestrator with context
        output = await self.orchestrator.process(
            text=text,
            speaker=speaker,
            conversation_context=context_str,
        )

        # Add to conversation context for future reference
        self.context.add_utterance(
            text=text,
            speaker=speaker,
            role=role,
            sentiment=output.sentiment,
            confidence=output.confidence,
            signals=output.signals,
            # Multi-agent outputs
            persona_name=output.persona_name,
            products_mentioned=output.products_mentioned,
            competitors_mentioned=output.competitors_mentioned,
            upsell_opportunities=output.upsell_opportunities,
        )

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
            # Multi-agent outputs
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
