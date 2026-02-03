#!/usr/bin/env python
"""Batch transcription using Azure Speech Service.

Uploads local audio files to Azure Blob Storage and transcribes them.
Optionally translates transcripts to English.

Usage:
    # Transcribe local files (uploads automatically)
    python batch_transcribe.py ./data/recordings

    # With speaker diarization
    python batch_transcribe.py ./data/recordings --diarization --max-speakers 4

    # Transcribe and translate to English
    python batch_transcribe.py ./data/recordings --translate --source-lang es

    # Save results to specific directory
    python batch_transcribe.py ./data/recordings --output ./transcripts

Environment variables required:
    AZURE_SPEECH_KEY - Your Azure Speech subscription key
    AZURE_SPEECH_REGION - Your Azure Speech region (e.g., swedencentral)
    AZURE_STORAGE_CONNECTION_STRING - Your Azure Storage connection string
    AZURE_TRANSLATOR_KEY - Your Azure Translator subscription key (for --translate)
    AZURE_TRANSLATOR_ENDPOINT - Your Azure Translator endpoint (for --translate)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from azure.storage.blob import (
    BlobServiceClient,
    ContainerSasPermissions,
    generate_container_sas,
)
from dotenv import load_dotenv

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac"}


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


def translate_document(
    input_file: Path,
    output_file: Path,
    target_lang: str,
    translator_key: str,
    translator_endpoint: str,
    source_lang: str | None = None,
) -> bool:
    """Translate a text file using Azure Translator.

    Args:
        input_file: Path to source text file.
        output_file: Path to save translated file.
        target_lang: Target language code (e.g., 'en').
        translator_key: Azure Translator subscription key.
        translator_endpoint: Azure Translator endpoint.
        source_lang: Source language code (auto-detect if None).

    Returns:
        True if successful, False otherwise.
    """
    url = f"{translator_endpoint}/translator/document:translate"

    headers = {
        "Ocp-Apim-Subscription-Key": translator_key,
    }

    params = {
        "targetLanguage": target_lang,
        "api-version": "2023-11-01-preview",
    }

    # Only set source language if specified (otherwise auto-detect)
    if source_lang:
        params["sourceLanguage"] = source_lang

    try:
        with open(input_file, "rb") as document:
            data = {
                "document": (input_file.name, document, "text/plain"),
            }
            response = requests.post(url, headers=headers, files=data, params=params)
            response.raise_for_status()

        with open(output_file, "wb") as out:
            out.write(response.content)

        return True

    except Exception as e:
        logger.error(f"Translation failed for {input_file.name}: {e}")
        return False


def translate_files(
    output_dir: Path,
    target_lang: str,
    translator_key: str,
    translator_endpoint: str,
    source_lang: str | None = None,
) -> None:
    """Translate all .txt files in output directory.

    Args:
        output_dir: Directory containing transcription files.
        target_lang: Target language code.
        translator_key: Azure Translator subscription key.
        translator_endpoint: Azure Translator endpoint.
        source_lang: Source language code (auto-detect if None).
    """
    # Search in output_dir and all subdirectories
    txt_files = list(output_dir.glob("**/*.txt"))
    # Exclude already translated files and diarized files
    txt_files = [
        f for f in txt_files
        if not f.name.endswith(f"_{target_lang}.txt") and not f.name.endswith("_diarized.txt")
    ]

    if not txt_files:
        logger.warning("No text files to translate")
        return

    lang_msg = f"from {source_lang}" if source_lang else "(auto-detect)"
    logger.info(f"Translating {len(txt_files)} files {lang_msg} to {target_lang}...")

    for txt_file in txt_files:
        output_file = txt_file.with_name(f"{txt_file.stem}_{target_lang}.txt")
        if translate_document(
            txt_file, output_file, target_lang,
            translator_key, translator_endpoint, source_lang
        ):
            logger.info(f"Translated: {txt_file.name} -> {output_file.name}")


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
    parser.add_argument("--output", type=Path, default=Path("./transcripts"), help="Output directory")
    parser.add_argument("--locale", type=str, default=None, help="Language locale (auto-detect if not set)")
    parser.add_argument("--diarization", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--min-speakers", type=int, default=1, help="Min speakers")
    parser.add_argument("--max-speakers", type=int, default=5, help="Max speakers")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")
    parser.add_argument("--translate", action="store_true", help="Translate transcripts to English")
    parser.add_argument("--translate-only", action="store_true", help="Only translate existing .txt files in output dir")
    parser.add_argument("--source-lang", type=str, default=None, help="Source language for translation (auto-detect if not set)")
    parser.add_argument("--target-lang", type=str, default="en", help="Target language for translation (default: en)")

    args = parser.parse_args()
    load_dotenv()

    # Get credentials
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION", "swedencentral")
    storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    translator_key = os.getenv("AZURE_TRANSLATOR_KEY")
    translator_endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")

    # Handle translate-only mode first
    if args.translate_only:
        if not translator_key or not translator_endpoint:
            logger.error("AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_ENDPOINT required")
            sys.exit(1)
        translate_files(
            args.output,
            args.target_lang,
            translator_key,
            translator_endpoint,
            source_lang=args.source_lang,
        )
        logger.info(f"Done! Translations saved to: {args.output}")
        sys.exit(0)

    if not speech_key:
        logger.error("AZURE_SPEECH_KEY not set")
        sys.exit(1)
    if args.translate and (not translator_key or not translator_endpoint):
        logger.error("AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_ENDPOINT required for --translate")
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

            transcription = transcriber.create_transcription(
                name=job_name,
                content_container_url=sas_uri,
                diarization=args.diarization,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                auto_detect_language=auto_detect,
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
            translate_files(
                args.output,
                args.target_lang,
                translator_key,
                translator_endpoint,
                source_lang=args.source_lang,  # None = auto-detect
            )

        logger.info(f"Done! Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
