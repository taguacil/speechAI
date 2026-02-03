"""Main entry point for Gemini transcription mode."""

import asyncio

from speechai.assistant_base import BaseSalesAssistant, run_assistant
from speechai.display import Colors, DisplayConfig
from speechai.transcription import TranscriptResult
from speechai.transcription_gemini import GeminiTranscriber


class GeminiSalesAssistant(BaseSalesAssistant):
    """Sales assistant using Gemini for transcription."""

    def __init__(self):
        super().__init__()
        self.transcriber = GeminiTranscriber()

    def _get_display_config(self) -> DisplayConfig:
        return DisplayConfig(
            mode_name="Sales Assistant - Gemini Mode",
            mode_color=Colors.MAGENTA,
            pipeline_description="Gemini STT → Parallel Agents → Consolidator",
        )

    def _get_stt_label(self) -> str:
        return "Gemini STT"

    def _get_mode_color(self) -> str:
        return Colors.MAGENTA

    def _start_transcription(self) -> None:
        self.transcriber.start(on_result=self._on_transcript)

    def _stop_transcription(self) -> None:
        self.transcriber.stop()

    def _on_transcript(self, result: TranscriptResult) -> None:
        """Handle transcript results."""
        if result.is_final and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._process_and_display(
                    text=result.text,
                    speaker=result.speaker_id,
                    stt_latency_ms=result.latency_ms,
                ),
                self._loop,
            )


def main() -> None:
    """Main entry point."""
    run_assistant(GeminiSalesAssistant())


if __name__ == "__main__":
    main()
