"""Azure Speech Service real-time transcription."""

import os
from collections.abc import Callable
from dataclasses import dataclass

import azure.cognitiveservices.speech as speechsdk


@dataclass
class TranscriptResult:
    """Result from transcription."""

    text: str
    is_final: bool  # True if this is a final result, False if interim


class AzureTranscriber:
    """Real-time transcription using Azure Speech Service.

    Supports both endpoint-based config (Azure AI Foundry) and region-based config.
    For continuous multi-utterance recognition, uses start_continuous_recognition().
    """

    def __init__(
        self,
        speech_key: str | None = None,
        speech_endpoint: str | None = None,
        language: str = "en-US",
    ):
        self.speech_key = speech_key or os.getenv("AZURE_SPEECH_KEY")
        self.speech_endpoint = speech_endpoint or os.getenv("AZURE_SPEECH_ENDPOINT")

        if not self.speech_key or not self.speech_endpoint:
            raise ValueError(
                "Azure Speech credentials required. "
                "Set AZURE_SPEECH_KEY and AZURE_SPEECH_ENDPOINT env vars."
            )

        self.language = language
        self._recognizer: speechsdk.SpeechRecognizer | None = None
        self._on_transcript: Callable[[TranscriptResult], None] | None = None

    def start(self, on_transcript: Callable[[TranscriptResult], None]) -> None:
        """Start continuous recognition from microphone.

        Uses start_continuous_recognition() for long-running multi-utterance
        recognition instead of recognize_once().

        Args:
            on_transcript: Callback called with each transcript result.
        """
        self._on_transcript = on_transcript

        # Use endpoint-based config (Azure AI Foundry style)
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            endpoint=self.speech_endpoint,
        )
        speech_config.speech_recognition_language = self.language

        # Use default microphone
        audio_config = speechsdk.AudioConfig(use_default_microphone=True)

        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        # Connect callbacks for continuous recognition
        self._recognizer.recognizing.connect(self._on_recognizing)
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.canceled.connect(self._on_canceled)
        self._recognizer.session_stopped.connect(self._on_session_stopped)

        # Start continuous recognition for multi-utterance
        self._recognizer.start_continuous_recognition()

    def stop(self) -> None:
        """Stop recognition."""
        if self._recognizer:
            self._recognizer.stop_continuous_recognition()
            self._recognizer = None

    def _on_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Handle interim results (while speaking)."""
        if evt.result.text and self._on_transcript:
            self._on_transcript(TranscriptResult(text=evt.result.text, is_final=False))

    def _on_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Handle final results (utterance complete)."""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            if evt.result.text and self._on_transcript:
                self._on_transcript(TranscriptResult(text=evt.result.text, is_final=True))

    def _on_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        """Handle cancellation/errors."""
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"Transcription error: {evt.error_details}")

    def _on_session_stopped(self, evt: speechsdk.SessionEventArgs) -> None:
        """Handle session stopped event."""
        pass  # Can add logging here if needed
