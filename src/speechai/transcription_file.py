"""File-based transcription for testing with recorded audio."""

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import openai

from speechai.transcription import TranscriptResult, TranscriptSegment


def _check_ffmpeg() -> None:
    """Check if ffmpeg is available."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found. Please install it:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: choco install ffmpeg"
        )


def convert_to_wav(input_path: Path, sample_rate: int = 16000) -> Path:
    """Convert audio file to WAV format using ffmpeg.

    Args:
        input_path: Path to input audio file (MP3, etc.)
        sample_rate: Target sample rate.

    Returns:
        Path to temporary WAV file.
    """
    _check_ffmpeg()
    output_path = Path(tempfile.mktemp(suffix=".wav"))

    subprocess.run(
        [
            "ffmpeg",
            "-i", str(input_path),
            "-ar", str(sample_rate),
            "-ac", "1",  # Mono
            "-sample_fmt", "s16",  # 16-bit
            "-y",  # Overwrite
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )

    return output_path


class FileTranscriberGemini:
    """Transcribe audio files using Gemini with speaker identification."""

    SYSTEM_PROMPT = """You are a transcription assistant for sales calls. Your task is to:
1. Transcribe the audio exactly
2. Identify who is speaking for each segment: "sales_rep" or "customer"

This is an OUTBOUND sales call, so the sales representative initiated the call.

Speaker identification guidelines:
- Sales rep: Usually introduces themselves, company, asks questions about needs, pitches products
- Customer: Responds to questions, asks about pricing/features, expresses concerns or interest

Respond with a JSON array of segments:
[
  {"speaker": "sales_rep", "text": "Hello, this is John from ABC company..."},
  {"speaker": "customer", "text": "Hi, yes I was looking at your products..."},
  ...
]"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.base_url = base_url or os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-vertex")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "sk-1234")

        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def transcribe(self, audio_path: Path) -> TranscriptResult | None:
        """Transcribe an audio file (returns combined transcript).

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)

        Returns:
            TranscriptResult with combined transcription.
        """
        results = list(self.stream(audio_path, lambda r: None))
        if not results:
            return None

        # Combine all segments
        combined_text = " ".join(r.text for r in results)
        total_latency = results[-1].latency_ms if results else 0

        return TranscriptResult(
            text=combined_text,
            is_final=True,
            speaker_id="Speaker",
            offset_ms=0,
            latency_ms=total_latency,
        )

    def stream(
        self,
        audio_path: Path,
        on_result: Callable[[TranscriptResult], None],
    ) -> list[TranscriptResult]:
        """Stream transcription results with speaker identification.

        Args:
            audio_path: Path to audio file.
            on_result: Callback for each transcription segment.

        Returns:
            List of all transcript results.
        """
        start = time.perf_counter()

        # Convert to WAV if needed
        if audio_path.suffix.lower() != ".wav":
            wav_path = convert_to_wav(audio_path)
            cleanup_wav = True
        else:
            wav_path = audio_path
            cleanup_wav = False

        results: list[TranscriptResult] = []

        try:
            # Read and encode audio
            with open(wav_path, "rb") as f:
                audio_data = f.read()

            wav_base64 = base64.b64encode(audio_data).decode("utf-8")

            # Send to Gemini with speaker identification prompt
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Transcribe this sales call audio with speaker identification. Return JSON array of segments.",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": wav_base64, "format": "wav"},
                            },
                        ],
                    },
                ],
                max_tokens=4000,
            )

            latency_ms = (time.perf_counter() - start) * 1000
            content = response.choices[0].message.content.strip()

            if not content:
                return results

            # Parse JSON response
            parsed_segments = self._parse_segments(content)

            # Build segment list and collect customer text
            transcript_segments = []
            customer_texts = []

            for segment in parsed_segments:
                speaker = segment.get("speaker", "sales_rep")
                text = segment.get("text", "").strip()

                if not text:
                    continue

                # Map speaker to consistent format
                speaker_id = "Speaker-1" if speaker == "sales_rep" else "Speaker-2"

                transcript_segments.append(TranscriptSegment(
                    text=text,
                    speaker_id=speaker_id,
                ))

                if speaker_id == "Speaker-2":
                    customer_texts.append(text)

            if not transcript_segments:
                return results

            # Create batched result if multiple segments, single otherwise
            if len(transcript_segments) == 1:
                result = TranscriptResult(
                    text=transcript_segments[0].text,
                    is_final=True,
                    speaker_id=transcript_segments[0].speaker_id,
                    offset_ms=0,
                    latency_ms=latency_ms,
                )
            else:
                # Combine customer text for analysis
                combined_text = " ".join(customer_texts) if customer_texts else transcript_segments[-1].text
                primary_speaker = "Speaker-2" if customer_texts else transcript_segments[-1].speaker_id

                result = TranscriptResult(
                    text=combined_text,
                    is_final=True,
                    speaker_id=primary_speaker,
                    offset_ms=0,
                    latency_ms=latency_ms,
                    segments=transcript_segments,
                )

            results.append(result)
            on_result(result)

        finally:
            if cleanup_wav and wav_path.exists():
                wav_path.unlink()

        return results

    def _parse_segments(self, content: str) -> list[dict]:
        """Parse JSON segments from Gemini response."""
        try:
            # Handle markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: return single segment with full text
            return [{"speaker": "sales_rep", "text": content}]


class FileTranscriberAzure:
    """Transcribe audio files using Azure Speech."""

    def __init__(
        self,
        speech_key: str | None = None,
        speech_endpoint: str | None = None,
        language: str = "en-US",
    ):
        import azure.cognitiveservices.speech as speechsdk

        self.speech_key = speech_key or os.getenv("AZURE_SPEECH_KEY")
        self.speech_endpoint = speech_endpoint or os.getenv("AZURE_SPEECH_ENDPOINT")
        self.language = language

        if not self.speech_key or not self.speech_endpoint:
            raise ValueError(
                "Azure Speech credentials required. "
                "Set AZURE_SPEECH_KEY and AZURE_SPEECH_ENDPOINT env vars."
            )

    def transcribe(self, audio_path: Path) -> TranscriptResult | None:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)

        Returns:
            TranscriptResult with transcription.
        """
        import azure.cognitiveservices.speech as speechsdk

        start = time.perf_counter()

        # Convert to WAV if needed
        if audio_path.suffix.lower() != ".wav":
            wav_path = convert_to_wav(audio_path)
            cleanup_wav = True
        else:
            wav_path = audio_path
            cleanup_wav = False

        try:
            # Configure speech
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                endpoint=self.speech_endpoint,
            )
            speech_config.speech_recognition_language = self.language

            # Use file as audio input
            audio_config = speechsdk.AudioConfig(filename=str(wav_path))

            # Use regular recognizer for file (not conversation transcriber)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )

            # Recognize all speech in file
            all_text = []
            done = False

            def on_recognized(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    all_text.append(evt.result.text)

            def on_session_stopped(evt):
                nonlocal done
                done = True

            def on_canceled(evt):
                nonlocal done
                done = True

            recognizer.recognized.connect(on_recognized)
            recognizer.session_stopped.connect(on_session_stopped)
            recognizer.canceled.connect(on_canceled)

            recognizer.start_continuous_recognition()

            # Wait for completion
            while not done:
                time.sleep(0.1)

            recognizer.stop_continuous_recognition()

            latency_ms = (time.perf_counter() - start) * 1000
            text = " ".join(all_text).strip()

            if not text:
                return None

            return TranscriptResult(
                text=text,
                is_final=True,
                speaker_id="Speaker",
                offset_ms=0,
                latency_ms=latency_ms,
            )

        finally:
            if cleanup_wav and wav_path.exists():
                wav_path.unlink()

    def stream(
        self,
        audio_path: Path,
        on_result: Callable[[TranscriptResult], None],
    ) -> None:
        """Stream transcription results as they are recognized.

        Uses ConversationTranscriber for speaker diarization.

        Args:
            audio_path: Path to audio file.
            on_result: Callback for each transcription result.
        """
        import azure.cognitiveservices.speech as speechsdk

        # Convert to WAV if needed
        if audio_path.suffix.lower() != ".wav":
            wav_path = convert_to_wav(audio_path)
            cleanup_wav = True
        else:
            wav_path = audio_path
            cleanup_wav = False

        # Speaker mapping for consistent labels
        speaker_map: dict[str, str] = {}
        speaker_count = 0

        def get_speaker_label(speaker_id: str) -> str:
            nonlocal speaker_count
            if not speaker_id or speaker_id == "Unknown":
                return "Unknown"
            if speaker_id not in speaker_map:
                speaker_count += 1
                speaker_map[speaker_id] = f"Speaker-{speaker_count}"
            return speaker_map[speaker_id]

        try:
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                endpoint=self.speech_endpoint,
            )
            speech_config.speech_recognition_language = self.language

            audio_config = speechsdk.AudioConfig(filename=str(wav_path))

            # Use ConversationTranscriber for speaker diarization
            transcriber = speechsdk.transcription.ConversationTranscriber(
                speech_config=speech_config,
                audio_config=audio_config,
            )

            done = False
            start_time = time.perf_counter()

            def on_transcribed(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    speaker = get_speaker_label(evt.result.speaker_id)
                    result = TranscriptResult(
                        text=evt.result.text,
                        is_final=True,
                        speaker_id=speaker,
                        offset_ms=evt.result.offset / 10000,  # ticks to ms
                        latency_ms=latency_ms,
                    )
                    on_result(result)

            def on_session_stopped(evt):
                nonlocal done
                done = True

            def on_canceled(evt):
                nonlocal done
                done = True

            transcriber.transcribed.connect(on_transcribed)
            transcriber.session_stopped.connect(on_session_stopped)
            transcriber.canceled.connect(on_canceled)

            transcriber.start_transcribing_async()

            while not done:
                time.sleep(0.1)

            transcriber.stop_transcribing_async()

        finally:
            if cleanup_wav and wav_path.exists():
                wav_path.unlink()


def load_audio_frames(
    audio_path: Path,
    sample_rate: int = 16000,
    frame_size: int = 480,
) -> tuple[np.ndarray, float]:
    """Load audio file and return as frames ready for streaming.

    Args:
        audio_path: Path to audio file.
        sample_rate: Target sample rate.
        frame_size: Samples per frame.

    Returns:
        Tuple of (audio frames as 2D numpy array, frame duration in seconds).
    """
    # Convert to WAV at correct sample rate
    wav_path = convert_to_wav(audio_path, sample_rate)

    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)

        # Convert to numpy array (16-bit mono)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Pad to multiple of frame_size
        remainder = len(audio_array) % frame_size
        if remainder:
            audio_array = np.pad(audio_array, (0, frame_size - remainder))

        # Reshape to (n_frames, frame_size) then add channel dim -> (n_frames, frame_size, 1)
        n_audio_frames = len(audio_array) // frame_size
        frames = audio_array.reshape(n_audio_frames, frame_size, 1)

        frame_duration_sec = frame_size / sample_rate

        return frames, frame_duration_sec

    finally:
        if wav_path.exists() and wav_path != audio_path:
            wav_path.unlink()
