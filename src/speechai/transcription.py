"""Azure Speech Service real-time transcription with speaker diarization."""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import azure.cognitiveservices.speech as speechsdk


@dataclass
class TranscriptSegment:
    """A single segment of transcription with speaker info."""

    text: str
    speaker_id: str  # "Speaker-1", "Speaker-2", etc.


@dataclass
class TranscriptResult:
    """Result from transcription with speaker info.

    For single-speaker results: text contains the transcript, speaker_id identifies speaker.
    For multi-speaker batches: segments contains individual parts, text is combined customer speech.
    """

    text: str
    is_final: bool
    speaker_id: str  # "Guest-1", "Guest-2", etc. or "Unknown"
    offset_ms: int  # When this was spoken (for ordering)
    latency_ms: float  # Time from speech start to final result
    # Optional: for batched multi-speaker results
    segments: list[TranscriptSegment] | None = None


class AzureTranscriber:
    """Real-time transcription with speaker diarization.

    Uses ConversationTranscriber for automatic speaker separation.
    Supports phrase list for domain-specific vocabulary.
    """

    def __init__(
        self,
        speech_key: str | None = None,
        speech_endpoint: str | None = None,
        language: str = "en-US",
        phrase_list: list[str] | None = None,
    ):
        self.speech_key = speech_key or os.getenv("AZURE_SPEECH_KEY")
        self.speech_endpoint = speech_endpoint or os.getenv("AZURE_SPEECH_ENDPOINT")

        if not self.speech_key or not self.speech_endpoint:
            raise ValueError(
                "Azure Speech credentials required. "
                "Set AZURE_SPEECH_KEY and AZURE_SPEECH_ENDPOINT env vars."
            )

        self.language = language
        self.phrase_list = phrase_list or []
        self._transcriber: speechsdk.transcription.ConversationTranscriber | None = None
        self._on_transcript: Callable[[TranscriptResult], None] | None = None

        # Speaker mapping: Azure IDs -> friendly names
        self._speaker_map: dict[str, str] = {}
        self._speaker_count = 0

        # Timing: track when speech starts for latency measurement
        self._speech_start_time: float | None = None

    def start(self, on_transcript: Callable[[TranscriptResult], None]) -> None:
        """Start continuous recognition with speaker diarization.

        Args:
            on_transcript: Callback called with each transcript result.
        """
        self._on_transcript = on_transcript

        # Speech config with endpoint
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            endpoint=self.speech_endpoint,
        )
        speech_config.speech_recognition_language = self.language

        # Enable detailed output for better accuracy
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        # Use default microphone
        audio_config = speechsdk.AudioConfig(use_default_microphone=True)

        # Create conversation transcriber for speaker diarization
        self._transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        # Add phrase list for better recognition of domain terms
        if self.phrase_list:
            phrase_list_grammar = speechsdk.PhraseListGrammar.from_recognizer(
                self._transcriber
            )
            for phrase in self.phrase_list:
                phrase_list_grammar.addPhrase(phrase)

        # Connect callbacks for diarized transcription
        self._transcriber.transcribing.connect(self._on_transcribing)
        self._transcriber.transcribed.connect(self._on_transcribed)
        self._transcriber.canceled.connect(self._on_canceled)
        self._transcriber.session_stopped.connect(self._on_session_stopped)

        # Start continuous transcription
        self._transcriber.start_transcribing_async()

    def stop(self) -> None:
        """Stop transcription."""
        if self._transcriber:
            self._transcriber.stop_transcribing_async()
            self._transcriber = None

    def _get_speaker_label(self, speaker_id: str) -> str:
        """Map Azure speaker ID to consistent label."""
        if not speaker_id or speaker_id == "Unknown":
            return "Unknown"

        if speaker_id not in self._speaker_map:
            self._speaker_count += 1
            # First speaker is usually customer in inbound, sales rep in outbound
            # Can be configured based on call direction
            self._speaker_map[speaker_id] = f"Speaker-{self._speaker_count}"

        return self._speaker_map[speaker_id]

    def _on_transcribing(
        self, evt: speechsdk.transcription.ConversationTranscriptionEventArgs
    ) -> None:
        """Handle interim results (while speaking)."""
        if evt.result.text and self._on_transcript:
            # Track speech start time for latency measurement
            if self._speech_start_time is None:
                self._speech_start_time = time.perf_counter()

            speaker = self._get_speaker_label(evt.result.speaker_id)
            offset = evt.result.offset // 10000  # Convert to ms

            self._on_transcript(
                TranscriptResult(
                    text=evt.result.text,
                    is_final=False,
                    speaker_id=speaker,
                    offset_ms=offset,
                    latency_ms=0,  # Not measured for interim
                )
            )

    def _on_transcribed(
        self, evt: speechsdk.transcription.ConversationTranscriptionEventArgs
    ) -> None:
        """Handle final results (utterance complete)."""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            if evt.result.text and self._on_transcript:
                # Calculate latency from speech start
                latency_ms = 0.0
                if self._speech_start_time is not None:
                    latency_ms = (time.perf_counter() - self._speech_start_time) * 1000
                    self._speech_start_time = None  # Reset for next utterance

                speaker = self._get_speaker_label(evt.result.speaker_id)
                offset = evt.result.offset // 10000

                self._on_transcript(
                    TranscriptResult(
                        text=evt.result.text,
                        is_final=True,
                        speaker_id=speaker,
                        offset_ms=offset,
                        latency_ms=latency_ms,
                    )
                )

    def _on_canceled(
        self, evt: speechsdk.transcription.ConversationTranscriptionCanceledEventArgs
    ) -> None:
        """Handle cancellation/errors."""
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"Transcription error: {evt.error_details}")

    def _on_session_stopped(self, evt: speechsdk.SessionEventArgs) -> None:
        """Handle session stopped."""
        pass

    def set_speaker_label(self, azure_speaker_id: str, label: str) -> None:
        """Manually set a speaker label (e.g., identify customer vs sales rep).

        Args:
            azure_speaker_id: The ID from Azure (e.g., "Guest-1")
            label: Friendly label (e.g., "Customer", "Sales Rep")
        """
        self._speaker_map[azure_speaker_id] = label

    def get_speakers(self) -> dict[str, str]:
        """Get current speaker mapping."""
        return self._speaker_map.copy()
