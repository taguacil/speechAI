#!/usr/bin/env python
"""Batch transcription using Azure Speech Service.

Uploads local audio files to Azure Blob Storage and transcribes them.
Optionally translates transcripts to formatted English.

Output directories:
    data/transcripts   - JSON files from Azure Speech
    data/translations  - Formatted English transcripts (_en.txt)
    data/phrases.txt   - Phrase list for improved recognition

Usage:
    # Transcribe local files
    python batch_transcribe.py ./data/recordings

    # Transcribe and translate (Azure Translator)
    python batch_transcribe.py ./data/recordings --translate

    # Transcribe and translate (LLM via LiteLLM)
    python batch_transcribe.py ./data/recordings --translate --llm

    # Translate existing JSON files only
    python batch_transcribe.py --translate-only
    python batch_transcribe.py --translate-only --llm

    # Download from existing job
    python batch_transcribe.py --job-id <id> --translate

    # Use existing container (skip upload)
    python batch_transcribe.py --container <name> --translate

Environment variables:
    AZURE_SPEECH_KEY - Azure Speech subscription key
    AZURE_SPEECH_REGION - Azure Speech region (e.g., swedencentral)
    AZURE_STORAGE_CONNECTION_STRING - Azure Storage connection string

    For Azure translation (default):
        AZURE_TRANSLATOR_KEY - Azure Translator subscription key
        AZURE_TRANSLATOR_ENDPOINT - Azure Translator endpoint

    For LLM translation (--llm):
        LLM_BASE_URL - LiteLLM proxy URL (default: http://localhost:4000)
        LLM_API_KEY - API key for LiteLLM
        LLM_MODEL_TRANSLATE - Model to use (default: gemini-2.5-pro)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import openai
import requests
from azure.storage.blob import (
    BlobServiceClient,
    ContainerSasPermissions,
    generate_container_sas,
)
from dotenv import load_dotenv

if TYPE_CHECKING:
    from typing import Union
    Translator = Union["AzureTranslator", "LLMTranslator"]

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac"}


def load_phrases(phrases_path: Path) -> list[str]:
    """Load phrases from a file (one phrase per line)."""
    if not phrases_path.exists():
        logger.warning(f"Phrases file not found: {phrases_path}")
        return []

    phrases = []
    with open(phrases_path, encoding="utf-8") as f:
        for line in f:
            phrase = line.strip()
            if phrase and not phrase.startswith("#"):
                phrases.append(phrase)

    logger.info(f"Loaded {len(phrases)} phrases from {phrases_path}")
    return phrases


def get_audio_files(paths: list[Path]) -> list[Path]:
    """Get all audio files from provided paths."""
    files = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
        elif path.is_dir():
            for ext in AUDIO_EXTENSIONS:
                files.extend(path.glob(f"*{ext}"))
    return sorted(files)


def get_container_sas_uri(connection_string: str, container_name: str) -> str:
    """Generate SAS URI for an existing container."""
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    account_name = blob_service.account_name
    account_key = blob_service.credential.account_key

    sas_token = generate_container_sas(
        account_name=account_name,
        container_name=container_name,
        account_key=account_key,
        permission=ContainerSasPermissions(read=True, list=True),
        expiry=datetime.now(timezone.utc) + timedelta(days=7),
    )

    return f"https://{account_name}.blob.core.windows.net/{container_name}?{sas_token}"


def upload_to_azure(
    connection_string: str,
    container_name: str,
    files: list[Path],
) -> str:
    """Upload files to Azure Blob Storage and return SAS URI."""
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)

    # Create container
    try:
        container_client.create_container()
        logger.info(f"Created container: {container_name}")
    except Exception:
        logger.info(f"Using existing container: {container_name}")

    # Upload files
    for file_path in files:
        blob_client = container_client.get_blob_client(file_path.name)
        logger.info(f"Uploading: {file_path.name}")
        with open(file_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)

    return get_container_sas_uri(connection_string, container_name)


class AzureBatchTranscriber:
    """Azure Batch Transcription client using REST API."""

    API_VERSION = "3.2"

    def __init__(self, subscription_key: str, region: str, locale: str = "en-US"):
        self.subscription_key = subscription_key
        self.region = region
        self.locale = locale
        self.base_url = f"https://{region}.api.cognitive.microsoft.com/speechtotext/v{self.API_VERSION}"
        self.headers = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Content-Type": "application/json",
        }

    def create_transcription(
        self,
        name: str,
        content_container_url: str,
        diarization: bool = False,
        min_speakers: int = 1,
        max_speakers: int = 5,
        auto_detect_language: bool = False,
        candidate_locales: list[str] | None = None,
        phrases: list[str] | None = None,
    ) -> dict:
        """Create a new batch transcription job."""
        url = f"{self.base_url}/transcriptions"

        properties = {
            "wordLevelTimestampsEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "None",
        }

        if diarization:
            properties["diarizationEnabled"] = True
            properties["diarization"] = {
                "speakers": {"minCount": min_speakers, "maxCount": max_speakers}
            }

        # Language auto-detection
        if auto_detect_language:
            # Default candidate languages: Indian languages
            locales = candidate_locales or [
                "en-IN",  # English (India)
                "gu-IN",  # Gujarati
                "hi-IN",  # Hindi
                "kn-IN",  # Kannada
            ]
            properties["languageIdentification"] = {
                "candidateLocales": locales,
                "mode": "Continuous"
            }

        # Phrase list for improved recognition
        if phrases:
            properties["customProperties"] = {
                "phraseList": phrases
            }
            logger.info(f"Added {len(phrases)} phrases to transcription request")

        body = {
            "displayName": name,
            "locale": self.locale,  # Always required, used as default
            "contentContainerUrl": content_container_url,
            "properties": properties,
        }

        response = requests.post(url, headers=self.headers, json=body)

        if response.status_code != 201:
            logger.error(f"API Error: {response.status_code}")
            logger.error(f"Response: {response.text}")
            response.raise_for_status()

        result = response.json()
        transcription_id = result["self"].split("/")[-1]
        logger.info(f"Created transcription job: {transcription_id}")
        return result

    def get_transcription(self, transcription_id: str) -> dict:
        """Get transcription job status."""
        url = f"{self.base_url}/transcriptions/{transcription_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_transcription_files(self, transcription_id: str) -> list[dict]:
        """Get result files for a completed transcription."""
        url = f"{self.base_url}/transcriptions/{transcription_id}/files"
        files = []
        while url:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            files.extend(data.get("values", []))
            url = data.get("@nextLink")
        return files

    def download_results(self, transcription_id: str) -> list[dict]:
        """Download transcription results."""
        files = self.get_transcription_files(transcription_id)
        results = []
        for file_info in files:
            if file_info.get("kind") != "Transcription":
                continue
            content_url = file_info["links"]["contentUrl"]
            response = requests.get(content_url)
            response.raise_for_status()
            results.append({"name": file_info["name"], "content": response.json()})
        return results

    def delete_transcription(self, transcription_id: str) -> None:
        """Delete a transcription job."""
        url = f"{self.base_url}/transcriptions/{transcription_id}"
        requests.delete(url, headers=self.headers)
        logger.info(f"Deleted transcription job: {transcription_id}")

    def wait_for_completion(
        self, transcription_id: str, poll_interval: int = 10, timeout: int = 3600
    ) -> dict:
        """Wait for transcription to complete."""
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Transcription did not complete within {timeout}s")

            transcription = self.get_transcription(transcription_id)
            status = transcription["status"]
            logger.info(f"Status: {status} (elapsed: {elapsed:.0f}s)")

            if status == "Succeeded":
                return transcription
            elif status == "Failed":
                error = transcription.get("properties", {}).get("error", {})
                raise RuntimeError(f"Transcription failed: {error.get('message')}")

            time.sleep(poll_interval)


class AzureTranslator:
    """Azure Translator for text segments."""

    def __init__(self, key: str, endpoint: str):
        self.key = key
        self.endpoint = endpoint

    def translate(self, text: str, target_lang: str = "en") -> str:
        """Translate text to target language."""
        if not text.strip():
            return text

        url = f"{self.endpoint}/translator/text/v3.0/translate"
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json",
        }
        params = {"api-version": "3.0", "to": target_lang}
        body = [{"text": text}]

        try:
            response = requests.post(url, headers=headers, params=params, json=body)
            response.raise_for_status()
            result = response.json()
            return result[0]["translations"][0]["text"]
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text


class LLMTranslator:
    """LLM-based translator using LiteLLM/OpenAI API."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def translate(self, text: str, target_lang: str = "en") -> str:
        """Translate text to target language using LLM."""
        if not text.strip():
            return text

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a translator. Translate the following text to {target_lang}. "
                                   "Output ONLY the translation, nothing else. Preserve the meaning and tone.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=500,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM translation failed: {e}")
            return text


def translate_json_to_transcript(
    json_file: Path,
    output_file: Path,
    translator: Translator | None = None,
) -> bool:
    """Convert Azure Speech JSON to formatted English transcript.

    Args:
        json_file: Path to Azure Speech JSON file.
        output_file: Path to save formatted transcript.
        translator: Translator instance for non-English content.

    Returns:
        True if successful, False otherwise.
    """
    try:
        with open(json_file) as f:
            data = json.load(f)

        phrases = data.get("recognizedPhrases", [])
        phrases = sorted(phrases, key=lambda p: p.get("offsetInTicks", 0))

        lines = []
        for phrase in phrases:
            channel = phrase.get("channel", 0)
            speaker = "Customer" if channel == 0 else "Sales Rep"

            n_best = phrase.get("nBest", [])
            if not n_best:
                continue

            display_text = n_best[0].get("display", "").strip()
            if not display_text:
                continue

            locale = phrase.get("locale", "en-US")
            offset_ms = phrase.get("offsetInTicks", 0) / 10_000  # ticks to ms
            offset_sec = offset_ms / 1000

            # Translate if not English
            if translator and not locale.startswith("en"):
                translated = translator.translate(display_text)
                lines.append(f"[{offset_sec:.1f}s] {speaker}: {translated}")
                if translated != display_text:
                    lines.append(f"         (Original {locale}: {display_text})")
            else:
                lines.append(f"[{offset_sec:.1f}s] {speaker}: {display_text}")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return True

    except Exception as e:
        logger.error(f"Failed to process {json_file.name}: {e}")
        return False


def translate_files(
    input_dir: Path,
    output_dir: Path,
    translator: AzureTranslator | LLMTranslator,
) -> None:
    """Translate all JSON files to formatted English transcripts.

    Args:
        input_dir: Directory containing JSON transcription files.
        output_dir: Directory to save translated transcripts.
        translator: Translator instance (Azure or LLM).
    """
    # Find JSON files (exclude analysis files)
    json_files = [
        f for f in input_dir.glob("**/*.json")
        if not f.name.endswith("_analysis.json")
    ]

    if not json_files:
        logger.warning("No JSON files to translate")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    translator_type = "LLM" if isinstance(translator, LLMTranslator) else "Azure"
    logger.info(f"Translating {len(json_files)} JSON files using {translator_type}...")

    for json_file in json_files:
        output_file = output_dir / f"{json_file.stem}_en.txt"
        if translate_json_to_transcript(json_file, output_file, translator):
            logger.info(f"Created: {output_file.name}")


def save_results(results: list[dict], output_dir: Path, diarized: bool = False) -> None:
    """Save transcription results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        content = result["content"]
        # Remove .json and any audio extension from the name
        # Also remove any directory prefix from Azure
        base_name = Path(result["name"]).name  # Get just filename
        for ext in [".json", ".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
            base_name = base_name.replace(ext, "")

        # Save raw JSON
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, "w") as f:
            json.dump(content, f, indent=2)

        # Save plain text
        combined = content.get("combinedRecognizedPhrases", [])
        if combined:
            text_content = "\n".join(p.get("display", "") for p in combined)
        else:
            phrases = content.get("recognizedPhrases", [])
            text_content = "\n".join(
                p.get("nBest", [{}])[0].get("display", "") for p in phrases
            )

        txt_path = output_dir / f"{base_name}.txt"
        with open(txt_path, "w") as f:
            f.write(text_content.strip())

        # Save diarized format
        if diarized:
            lines = []
            for phrase in content.get("recognizedPhrases", []):
                speaker = phrase.get("speaker", 0)
                best = phrase.get("nBest", [{}])[0]
                phrase_text = best.get("display", "")
                offset = phrase.get("offsetInTicks", 0) / 10_000_000
                if phrase_text:
                    lines.append(f"[{offset:.2f}s] Speaker-{speaker}: {phrase_text}")

            diarized_path = output_dir / f"{base_name}_diarized.txt"
            with open(diarized_path, "w") as f:
                f.write("\n".join(lines))

        logger.info(f"Saved: {base_name}")


def main():
    parser = argparse.ArgumentParser(description="Batch transcription with Azure Speech")
    parser.add_argument("input", type=Path, nargs="*", help="Audio files or directories (skip if using --container)")
    parser.add_argument("--container", type=str, help="Use existing container name (skip upload)")
    parser.add_argument("--job-id", type=str, help="Download results from existing transcription job (skip transcription)")
    parser.add_argument("--output", type=Path, default=Path("./data/transcripts"), help="Output directory for JSON transcripts")
    parser.add_argument("--translations-dir", type=Path, default=Path("./data/translations"), help="Output directory for translations")
    parser.add_argument("--locale", type=str, default=None, help="Language locale (auto-detect if not set)")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--min-speakers", type=int, default=1, help="Min speakers")
    parser.add_argument("--max-speakers", type=int, default=5, help="Max speakers")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument("--translate", action="store_true", help="Translate JSON files to English transcripts")
    parser.add_argument("--translate-only", action="store_true", help="Only translate existing JSON files in output dir")
    parser.add_argument("--llm", action="store_true", help="Use LLM for translation instead of Azure Translator")
    parser.add_argument("--phrases", type=Path, default=Path("./data/phrases.txt"), help="File with phrases to boost recognition (one per line)")

    args = parser.parse_args()
    load_dotenv()

    # Get credentials
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION", "swedencentral")
    storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    translator_key = os.getenv("AZURE_TRANSLATOR_KEY")
    translator_endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:4000")
    llm_api_key = os.getenv("LLM_API_KEY", "sk-1234")
    llm_model = os.getenv("LLM_MODEL_TRANSLATE", "gemini-2.5-pro")

    def get_translator() -> AzureTranslator | LLMTranslator:
        """Create appropriate translator based on --llm flag."""
        if args.llm:
            logger.info(f"Using LLM translator: {llm_model}")
            return LLMTranslator(llm_base_url, llm_api_key, llm_model)
        else:
            if not translator_key or not translator_endpoint:
                logger.error("AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_ENDPOINT required (or use --llm)")
                sys.exit(1)
            logger.info("Using Azure Translator")
            return AzureTranslator(translator_key, translator_endpoint)

    # Handle translate-only mode first
    if args.translate_only:
        translator = get_translator()
        translate_files(args.output, args.translations_dir, translator)
        logger.info(f"Done! Translations saved to: {args.translations_dir}")
        sys.exit(0)

    if not speech_key:
        logger.error("AZURE_SPEECH_KEY not set")
        sys.exit(1)

    # Either download from existing job, use existing container, or upload files
    sas_uri = None
    if args.job_id:
        # Just downloading, no container needed
        logger.info(f"Will download from existing job: {args.job_id}")
    elif args.container:
        if not storage_conn:
            logger.error("AZURE_STORAGE_CONNECTION_STRING not set")
            sys.exit(1)
        logger.info(f"Using existing container: {args.container}")
        sas_uri = get_container_sas_uri(storage_conn, args.container)
    else:
        if not args.input:
            logger.error("Provide input files, --container, or --job-id")
            sys.exit(1)
        if not storage_conn:
            logger.error("AZURE_STORAGE_CONNECTION_STRING not set")
            sys.exit(1)

        files = get_audio_files(args.input)
        if not files:
            logger.error("No audio files found")
            sys.exit(1)

        logger.info(f"Found {len(files)} audio files")
        container_name = f"transcribe-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logger.info("Uploading files to Azure Blob Storage...")
        sas_uri = upload_to_azure(storage_conn, container_name, files)
        logger.info("Upload complete")

    # Create transcriber
    locale = args.locale or "en-IN"  # Default for transcriber init
    transcriber = AzureBatchTranscriber(speech_key, speech_region, locale)

    try:
        # If job-id provided, just download results
        if args.job_id:
            logger.info(f"Downloading results from job: {args.job_id}")
            results = transcriber.download_results(args.job_id)
            save_results(results, args.output, diarized=args.diarization)
        else:
            # Run new transcription
            job_name = f"batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            auto_detect = args.locale is None
            if auto_detect:
                logger.info("Language auto-detection enabled")

            # Load phrases if file exists
            phrases = load_phrases(args.phrases) if args.phrases.exists() else []

            transcription = transcriber.create_transcription(
                name=job_name,
                content_container_url=sas_uri,
                diarization=args.diarization,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                auto_detect_language=auto_detect,
                phrases=phrases,
            )
            transcription_id = transcription["self"].split("/")[-1]

            logger.info("Waiting for transcription...")
            transcriber.wait_for_completion(transcription_id, timeout=args.timeout)

            logger.info("Downloading results...")
            results = transcriber.download_results(transcription_id)
            save_results(results, args.output, diarized=args.diarization)

            transcriber.delete_transcription(transcription_id)

        # Translate if requested
        if args.translate:
            translator = get_translator()
            translate_files(args.output, args.translations_dir, translator)
            logger.info(f"Translations saved to: {args.translations_dir}")

        logger.info(f"Done! Transcripts saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
