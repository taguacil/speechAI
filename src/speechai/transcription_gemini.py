"""Gemini-based transcription with native audio input.

Uses Gemini 2.0 Flash via LiteLLM for transcription.
"""

import base64
import io
import json
import os
import threading
import time
import wave
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd
import webrtcvad

import openai

from speechai.transcription import TranscriptResult, TranscriptSegment


class GeminiTranscriber:
    """Real-time transcription using Gemini with native audio.

    Uses Voice Activity Detection (VAD) to buffer audio during speech
    and sends chunks to Gemini when silence is detected.
    """

    # Audio settings (WebRTC VAD requires specific formats)
    SAMPLE_RATE = 16000
    CHANNELS = 1
    FRAME_DURATION_MS = 30
    FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

    # VAD settings
    VAD_MODE = 3
    SILENCE_THRESHOLD_MS = 500
    MIN_SPEECH_MS = 300
    MAX_BUFFER_MS = 10000

    # Prompt for transcription with speaker identification
    # Supports multiple speakers per audio chunk
    SYSTEM_PROMPT = """You are a transcription assistant for sales calls. Your task is to:
1. Transcribe the audio exactly
2. Identify who is speaking for each segment: "sales_rep" or "customer"

This is an OUTBOUND sales call, so the sales representative initiated the call.

Speaker identification guidelines:
- Sales rep: Introduces themselves, company name, asks questions about needs, pitches products, uses professional tone
- Customer: Responds to questions, asks about pricing/features, expresses concerns or interest

IMPORTANT: The audio may contain MULTIPLE speakers. If you hear speaker changes, return an array of segments.

Response format - use array if multiple speakers, single object if one speaker:
- Single speaker: {"transcript": "...", "speaker": "sales_rep" or "customer"}
- Multiple speakers: [{"transcript": "...", "speaker": "sales_rep"}, {"transcript": "...", "speaker": "customer"}]"""

    USER_PROMPT_WITH_CONTEXT = """Recent conversation:
{context}

Transcribe this audio. If multiple people speak, return an array of segments in order.
Respond with JSON only (single object or array)."""

    USER_PROMPT_NO_CONTEXT = """This is the start of a sales call. Transcribe the audio.
If multiple people speak, return an array of segments in order.
Respond with JSON only (single object or array)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        debug: bool | None = None,
    ):
        self.base_url = base_url or os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-vertex")
        self.api_key = api_key or os.getenv("LITELLM_API_KEY", "sk-1234")
        if debug is None:
            self.debug = os.getenv("SPEECHAI_DEBUG", "").lower() in ("1", "true", "yes")
        else:
            self.debug = debug

        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self._vad = webrtcvad.Vad(self.VAD_MODE)
        self._audio_buffer: list[bytes] = []
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0

        self._on_result: Callable[[TranscriptResult], None] | None = None
        self._running = False
        self._stream: sd.InputStream | None = None

        # Conversation context for speaker identification
        self._recent_transcripts: list[tuple[str, str]] = []  # (speaker, text)
        self._max_context_items = 5
        self._context_lock = threading.Lock()

        # Serialization: only one Gemini call at a time
        self._processing = False
        self._processing_lock = threading.Lock()
        self._pending_audio: bytes | None = None

    def start(
        self,
        on_result: Callable[[TranscriptResult], None],
        start_mic: bool = True,
    ) -> None:
        """Start listening and processing audio.

        Args:
            on_result: Callback for transcription results.
            start_mic: If True, start microphone capture. Set to False for file mode
                       where audio frames are fed directly via _audio_callback.
        """
        self._on_result = on_result
        self._running = True
        self._audio_buffer = []
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._recent_transcripts = []
        self._processing = False
        self._pending_audio = None

        if start_mic:
            self._stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=np.int16,
                blocksize=self.FRAME_SIZE,
                callback=self._audio_callback,
            )
            self._stream.start()

    def stop(self) -> None:
        """Stop listening."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Process incoming audio frames."""
        if not self._running:
            return

        audio_bytes = indata.tobytes()

        try:
            is_speech = self._vad.is_speech(audio_bytes, self.SAMPLE_RATE)
        except Exception:
            is_speech = False

        if is_speech:
            self._silence_frames = 0
            self._speech_frames += 1
            self._is_speaking = True
            self._audio_buffer.append(audio_bytes)
        else:
            if self._is_speaking:
                self._silence_frames += 1
                self._audio_buffer.append(audio_bytes)

                silence_ms = self._silence_frames * self.FRAME_DURATION_MS
                speech_ms = self._speech_frames * self.FRAME_DURATION_MS

                if silence_ms >= self.SILENCE_THRESHOLD_MS and speech_ms >= self.MIN_SPEECH_MS:
                    self._process_buffer()

        buffer_ms = len(self._audio_buffer) * self.FRAME_DURATION_MS
        if buffer_ms >= self.MAX_BUFFER_MS:
            self._process_buffer()

    def _process_buffer(self) -> None:
        """Process buffered audio through Gemini.

        Serializes API calls - if a call is in progress, audio is queued
        and processed when the current call completes.
        """
        if not self._audio_buffer or not self._on_result:
            return

        buffer_len = len(self._audio_buffer)
        buffer_ms = buffer_len * self.FRAME_DURATION_MS
        audio_data = b"".join(self._audio_buffer)
        self._audio_buffer = []
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0

        with self._processing_lock:
            if self._processing:
                # API call in progress - queue this audio
                if self._pending_audio:
                    self._pending_audio += audio_data
                    if self.debug:
                        print(f"\n[DEBUG] Queued buffer: {buffer_len} frames ({buffer_ms}ms) - appended to pending")
                else:
                    self._pending_audio = audio_data
                    if self.debug:
                        print(f"\n[DEBUG] Queued buffer: {buffer_len} frames ({buffer_ms}ms) - new pending")
                return

            # No call in progress - start one
            self._processing = True
            if self.debug:
                print(f"\n[DEBUG] Processing buffer: {buffer_len} frames ({buffer_ms}ms, {len(audio_data)} bytes)")

        thread = threading.Thread(
            target=self._send_to_gemini_serialized,
            args=(audio_data,),
            daemon=True,
        )
        thread.start()

    def _build_context_prompt(self) -> str:
        """Build the user prompt with conversation context."""
        with self._context_lock:
            if not self._recent_transcripts:
                return self.USER_PROMPT_NO_CONTEXT

            context_lines = []
            for speaker, text in self._recent_transcripts:
                label = "Rep" if speaker == "sales_rep" else "Customer"
                context_lines.append(f"{label}: {text}")

        context = "\n".join(context_lines)
        return self.USER_PROMPT_WITH_CONTEXT.format(context=context)

    def _send_to_gemini_serialized(self, audio_data: bytes) -> None:
        """Send audio to Gemini, then process any pending audio."""
        try:
            self._send_to_gemini(audio_data)
        finally:
            # Check for pending audio
            with self._processing_lock:
                if self._pending_audio:
                    next_audio = self._pending_audio
                    self._pending_audio = None
                    if self.debug:
                        pending_ms = len(next_audio) // 2 // self.SAMPLE_RATE * 1000
                        print(f"\n[DEBUG] Processing pending audio: {len(next_audio)} bytes (~{pending_ms}ms)")
                else:
                    self._processing = False
                    return

            # Process pending audio (still in this thread)
            self._send_to_gemini_serialized(next_audio)

    def _send_to_gemini(self, audio_data: bytes) -> None:
        """Send audio to Gemini and process response."""
        start = time.perf_counter()

        try:
            wav_base64 = self._audio_to_wav_base64(audio_data)
            user_prompt = self._build_context_prompt()

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
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": wav_base64, "format": "wav"},
                            },
                        ],
                    },
                ],
                max_tokens=300,
            )

            latency_ms = (time.perf_counter() - start) * 1000
            results = self._parse_response(response, latency_ms)
            if self.debug:
                print(f"[DEBUG] Gemini returned {len(results)} result(s) in {latency_ms:.0f}ms")

            if self._on_result and results:
                # Add all segments to context for future speaker identification
                with self._context_lock:
                    for result in results:
                        speaker = "sales_rep" if result.speaker_id == "Speaker-1" else "customer"
                        self._recent_transcripts.append((speaker, result.text))
                        if len(self._recent_transcripts) > self._max_context_items:
                            self._recent_transcripts.pop(0)

                # Create batched result for processing
                if len(results) == 1:
                    # Single segment - send as-is
                    if self.debug:
                        print(f"[DEBUG] Calling callback with single result: {results[0].text[:50]}...")
                    self._on_result(results[0])
                else:
                    # Multiple segments - combine into batch
                    segments = [
                        TranscriptSegment(text=r.text, speaker_id=r.speaker_id)
                        for r in results
                    ]
                    # Combine customer text for analysis (skip sales rep)
                    customer_texts = [
                        r.text for r in results if r.speaker_id == "Speaker-2"
                    ]
                    combined_text = " ".join(customer_texts) if customer_texts else results[-1].text

                    # Use the last customer speaker or last speaker overall
                    primary_speaker = "Speaker-2" if customer_texts else results[-1].speaker_id

                    batch_result = TranscriptResult(
                        text=combined_text,
                        is_final=True,
                        speaker_id=primary_speaker,
                        offset_ms=0,
                        latency_ms=latency_ms,
                        segments=segments,
                    )
                    if self.debug:
                        print(f"[DEBUG] Calling callback with batch ({len(segments)} segments): {combined_text[:50]}...")
                    self._on_result(batch_result)

        except Exception as e:
            # Only log if there's a meaningful error message
            error_msg = str(e).strip()
            if error_msg:
                print(f"\n[Gemini] Error: {error_msg}")

    def _audio_to_wav_base64(self, audio_data: bytes) -> str:
        """Convert raw PCM audio to base64-encoded WAV."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.SAMPLE_RATE)
            wav_file.writeframes(audio_data)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _parse_response(self, response: Any, latency_ms: float) -> list[TranscriptResult]:
        """Parse Gemini transcription response with speaker identification.

        Returns a list of TranscriptResults (may be multiple if audio contains
        multiple speakers).
        """
        try:
            content = response.choices[0].message.content
            if not content:
                return []

            content = content.strip()
            if not content:
                return []

            # Try to parse as JSON
            try:
                # Handle markdown code blocks if present
                if content.startswith("```"):
                    parts = content.split("```")
                    if len(parts) >= 2:
                        content = parts[1]
                        if content.startswith("json"):
                            content = content[4:]
                        content = content.strip()

                data = json.loads(content)

                # Handle array of segments (multiple speakers)
                if isinstance(data, list):
                    results = []
                    for segment in data:
                        text = segment.get("transcript", "").strip()
                        speaker = segment.get("speaker", "sales_rep")
                        if text:
                            speaker_id = "Speaker-1" if speaker == "sales_rep" else "Speaker-2"
                            results.append(TranscriptResult(
                                text=text,
                                is_final=True,
                                speaker_id=speaker_id,
                                offset_ms=0,
                                latency_ms=latency_ms,
                            ))
                    return results

                # Handle single object (one speaker)
                text = data.get("transcript", "").strip()
                speaker = data.get("speaker", "sales_rep")

                if not text:
                    return []

                # Map speaker to consistent format
                speaker_id = "Speaker-1" if speaker == "sales_rep" else "Speaker-2"

                return [TranscriptResult(
                    text=text,
                    is_final=True,
                    speaker_id=speaker_id,
                    offset_ms=0,
                    latency_ms=latency_ms,
                )]
            except json.JSONDecodeError:
                # Fallback: try to extract text if it looks like partial JSON
                import re
                text = content
                if '{"transcript"' in content or '"transcript"' in content:
                    # Try to extract the transcript value
                    match = re.search(r'"transcript"\s*:\s*"([^"]*)"', content)
                    if match:
                        text = match.group(1)

                # Clean up any remaining JSON artifacts
                text = text.strip()
                if not text or text.startswith("{") or text.startswith("["):
                    return []

                return [TranscriptResult(
                    text=text,
                    is_final=True,
                    speaker_id="Speaker-1" if not self._recent_transcripts else "Speaker-2",
                    offset_ms=0,
                    latency_ms=latency_ms,
                )]
        except (AttributeError, IndexError):
            # No valid response - likely silence or noise
            return []
