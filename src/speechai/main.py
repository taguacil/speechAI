"""Main entry point for Azure Speech mode."""

import asyncio

from speechai.assistant_base import BaseSalesAssistant, run_assistant
from speechai.display import Colors, DisplayConfig
from speechai.transcription import AzureTranscriber, TranscriptResult


class AzureSalesAssistant(BaseSalesAssistant):
    """Sales assistant using Azure Speech for transcription."""

    def __init__(self):
        super().__init__()
        phrase_list = self.prompts.get("phrase_list", [])
        self.transcriber = AzureTranscriber(phrase_list=phrase_list)

    def _get_display_config(self) -> DisplayConfig:
        return DisplayConfig(
            mode_name="Sales Assistant - Azure Mode",
            mode_color=Colors.BLUE,
            pipeline_description="Azure STT → Parallel Agents → Consolidator",
        )

    def _get_stt_label(self) -> str:
        return "Azure STT"

    def _get_mode_color(self) -> str:
        return Colors.BLUE

    def _start_transcription(self) -> None:
        self.transcriber.start(on_transcript=self._on_transcript)

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
        elif not result.is_final:
            self._show_interim(result.speaker_id, result.text)


def main() -> None:
    """Main entry point."""
    run_assistant(AzureSalesAssistant())


if __name__ == "__main__":
    main()
