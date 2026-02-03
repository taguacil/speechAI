"""File-based transcription for testing with recorded audio."""

import base64
import io
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import openai

from speechai.transcription import TranscriptResult


def convert_to_wav(input_path: Path, sample_rate: int = 16000) -> Path:
    """Convert audio file to WAV format using ffmpeg.

    Args:
        input_path: Path to input audio file (MP3, etc.)
        sample_rate: Target sample rate.

    Returns:
        Path to temporary WAV file.
    """
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
    """Transcribe audio files using Gemini."""

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
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)

        Returns:
            TranscriptResult with transcription.
        """
        start = time.perf_counter()

        # Convert to WAV if needed
        if audio_path.suffix.lower() != ".wav":
            wav_path = convert_to_wav(audio_path)
            cleanup_wav = True
        else:
            wav_path = audio_path
            cleanup_wav = False

        try:
            # Read and encode audio
            with open(wav_path, "rb") as f:
                audio_data = f.read()

            wav_base64 = base64.b64encode(audio_data).decode("utf-8")

            # Send to Gemini
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Transcribe this audio exactly. Respond with ONLY the transcription text.",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": wav_base64, "format": "wav"},
                            },
                        ],
                    },
                ],
                max_tokens=1000,
            )

            latency_ms = (time.perf_counter() - start) * 1000
            text = response.choices[0].message.content.strip()

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
