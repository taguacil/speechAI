#!/usr/bin/env python
"""LLM-based analysis of transcripts.

Reads Azure Speech JSON files, translates non-English content, and analyzes.

Usage:
    # Analyze all transcripts in a directory
    python analyze_transcripts.py ./data/transcripts

    # Use specific model
    python analyze_transcripts.py ./data/transcripts --model gemini-2.5-pro

    # Output report to specific file
    python analyze_transcripts.py ./data/transcripts --output ./reports/analysis.md

Environment variables:
    LLM_BASE_URL - LiteLLM proxy URL (default: http://localhost:4000)
    LLM_API_KEY - API key for LiteLLM
    LLM_MODEL - Default model to use
    AZURE_TRANSLATOR_KEY - Azure Translator key (for translation)
    AZURE_TRANSLATOR_ENDPOINT - Azure Translator endpoint
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import openai
import requests
from dotenv import load_dotenv

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


INDIVIDUAL_ANALYSIS_PROMPT = """Analyze this sales call transcript and extract comprehensive information.

TRANSCRIPT:
{transcript}

Provide a detailed analysis in the following JSON format:
{{
    "summary": "2-3 sentence summary of the call",
    "caller_type": "new lead | existing customer | inquiry | complaint | other",
    "caller_intent": "what the caller wanted to achieve",
    "sentiment": {{
        "overall": "positive | negative | neutral | mixed",
        "progression": "improved | worsened | stable",
        "key_moments": ["moment 1", "moment 2"]
    }},
    "key_topics": ["topic1", "topic2"],
    "products_services_mentioned": ["product1", "service1"],
    "objections_raised": [
        {{"objection": "description", "handled": true, "resolution": "how it was addressed"}}
    ],
    "questions_asked": ["question1", "question2"],
    "commitments_made": ["commitment1", "commitment2"],
    "action_items": ["action1", "action2"],
    "call_outcome": "sale | appointment | callback | lost | unresolved",
    "sales_rep_performance": {{
        "strengths": ["strength1", "strength2"],
        "areas_for_improvement": ["area1", "area2"]
    }},
    "customer_insights": {{
        "pain_points": ["pain1", "pain2"],
        "priorities": ["priority1", "priority2"],
        "decision_factors": ["factor1", "factor2"]
    }},
    "follow_up_required": true,
    "follow_up_notes": "what follow-up is needed",
    "notable_quotes": ["quote1", "quote2"],
    "risk_flags": ["any concerning patterns or red flags"]
}}

Respond with ONLY valid JSON, no other text."""


COMBINED_REPORT_PROMPT = """You are analyzing a batch of {count} sales call transcripts. Here are the individual analyses:

{analyses}

Create a comprehensive executive report that synthesizes all findings. Include:

# Executive Summary
- Total calls analyzed
- Overall performance metrics
- Key trends and patterns

# Call Outcomes Breakdown
- Distribution of outcomes (sales, appointments, lost, etc.)
- Success rate analysis

# Customer Insights
- Common pain points across calls
- Frequently mentioned products/services
- Decision factors patterns

# Objection Analysis
- Most common objections
- How well objections were handled
- Recommended responses

# Sales Team Performance
- Common strengths
- Areas needing improvement
- Best practices observed

# Risk Flags & Concerns
- Patterns that need attention
- Recurring issues

# Recommendations
- Top 3-5 actionable recommendations based on the analysis

# Individual Call Summaries
Brief summary of each call with outcome

Format the report in clean Markdown."""


class Translator:
    """Azure Translator for non-English content."""

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

        params = {
            "api-version": "3.0",
            "to": target_lang,
        }

        body = [{"text": text}]

        try:
            response = requests.post(url, headers=headers, params=params, json=body)
            response.raise_for_status()
            result = response.json()
            return result[0]["translations"][0]["text"]
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text  # Return original if translation fails


def extract_conversation_from_json(json_data: dict, translator: Translator | None = None) -> str:
    """Extract and format conversation from Azure Speech JSON.

    Args:
        json_data: Parsed Azure Speech JSON.
        translator: Optional translator for non-English content.

    Returns:
        Formatted conversation string.
    """
    lines = []

    # Get recognized phrases with timing and channel info
    phrases = json_data.get("recognizedPhrases", [])

    # Sort by offset time
    phrases = sorted(phrases, key=lambda p: p.get("offsetInTicks", 0))

    for phrase in phrases:
        channel = phrase.get("channel", 0)
        speaker = "Customer" if channel == 0 else "Sales Rep"

        # Get best transcription
        n_best = phrase.get("nBest", [])
        if not n_best:
            continue

        display_text = n_best[0].get("display", "").strip()
        if not display_text:
            continue

        locale = phrase.get("locale", "en-US")
        offset_ms = phrase.get("offsetMilliseconds", 0)
        offset_sec = offset_ms / 1000

        # Translate if not English and translator available
        if translator and not locale.startswith("en"):
            translated = translator.translate(display_text)
            lines.append(f"[{offset_sec:.1f}s] {speaker}: {translated}")
            if translated != display_text:
                lines.append(f"         (Original {locale}: {display_text})")
        else:
            lines.append(f"[{offset_sec:.1f}s] {speaker}: {display_text}")

    return "\n".join(lines)


class TranscriptAnalyzer:
    """Analyzes transcripts using LLM."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def analyze_single(self, conversation: str, filename: str) -> dict:
        """Analyze a single transcript."""
        logger.info(f"Analyzing: {filename} ({len(conversation)} chars)")

        if not conversation.strip():
            logger.warning(f"Empty transcript: {filename}")
            return {"filename": filename, "error": "Empty transcript"}

        try:
            logger.info(f"Sending to LLM ({self.model})...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": INDIVIDUAL_ANALYSIS_PROMPT.format(transcript=conversation),
                    }
                ],
                max_tokens=4000,
                temperature=0.1,
            )
            logger.info(f"Received response for: {filename}")

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            analysis = json.loads(content)
            analysis["filename"] = filename
            return analysis

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for {filename}: {e}")
            return {
                "filename": filename,
                "error": "Failed to parse analysis",
                "raw_response": content if 'content' in locals() else None,
            }
        except Exception as e:
            logger.error(f"Error analyzing {filename}: {e}")
            import traceback
            traceback.print_exc()
            return {"filename": filename, "error": str(e)}

    def generate_combined_report(self, analyses: list[dict]) -> str:
        """Generate combined report from all analyses."""
        logger.info("Generating combined report...")

        # Format analyses for prompt
        analyses_text = ""
        for i, analysis in enumerate(analyses, 1):
            analyses_text += f"\n--- Call {i}: {analysis.get('filename', 'Unknown')} ---\n"
            analyses_text += json.dumps(analysis, indent=2, ensure_ascii=False)
            analyses_text += "\n"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": COMBINED_REPORT_PROMPT.format(
                            count=len(analyses),
                            analyses=analyses_text,
                        ),
                    }
                ],
                max_tokens=8000,
                temperature=0.2,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"# Error Generating Report\n\nError: {e}"


def get_json_files(input_path: Path) -> list[Path]:
    """Get all JSON transcript files."""
    if input_path.is_file() and input_path.suffix == ".json":
        return [input_path]

    return sorted(input_path.glob("**/*.json"))


def main():
    parser = argparse.ArgumentParser(description="Analyze transcripts with LLM")
    parser.add_argument("input", type=Path, help="JSON transcript file or directory")
    parser.add_argument("--output", type=Path, default=None, help="Output report file")
    parser.add_argument("--model", type=str, default=None, help="LLM model to use")
    parser.add_argument("--save-json", action="store_true", help="Save individual analyses as JSON")
    parser.add_argument("--save-transcripts", action="store_true", help="Save English transcripts")
    parser.add_argument("--no-translate", action="store_true", help="Skip translation")

    args = parser.parse_args()
    load_dotenv()

    # Get LLM config
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:4000")
    api_key = os.getenv("LLM_API_KEY", "sk-1234")
    model = args.model or os.getenv("LLM_MODEL", "gemini-2.5-pro")

    # Get translator config
    translator = None
    if not args.no_translate:
        translator_key = os.getenv("AZURE_TRANSLATOR_KEY")
        translator_endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
        if translator_key and translator_endpoint:
            translator = Translator(translator_key, translator_endpoint)
            logger.info("Translation enabled")
        else:
            logger.warning("Translation disabled (AZURE_TRANSLATOR_KEY/ENDPOINT not set)")

    logger.info(f"Using model: {model}")

    # Get JSON files
    files = get_json_files(args.input)
    if not files:
        logger.error("No JSON transcript files found")
        sys.exit(1)

    logger.info(f"Found {len(files)} JSON files")

    # Initialize analyzer
    analyzer = TranscriptAnalyzer(base_url, api_key, model)

    # Process each file
    analyses = []
    for file_path in files:
        logger.info(f"Processing: {file_path.name}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            continue

        # Extract and translate conversation
        conversation = extract_conversation_from_json(json_data, translator)

        if conversation.strip():
            # Save English transcript if requested
            if args.save_transcripts:
                transcript_dir = args.output.parent if args.output else Path(".")
                transcript_path = transcript_dir / f"{file_path.stem}_en.txt"
                transcript_path.write_text(conversation, encoding="utf-8")
                logger.info(f"Saved transcript: {transcript_path}")

            analysis = analyzer.analyze_single(conversation, file_path.stem)
            analyses.append(analysis)
        else:
            logger.warning(f"Empty conversation in: {file_path.name}")

    logger.info(f"Analyzed {len(analyses)} transcripts")

    if not analyses:
        logger.error("No transcripts were analyzed successfully")
        sys.exit(1)

    # Save individual analyses as JSON if requested
    if args.save_json:
        json_output = args.output.with_suffix(".json") if args.output else Path("analyses.json")
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(analyses, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved individual analyses to: {json_output}")

    # Generate combined report
    report = analyzer.generate_combined_report(analyses)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = Path(f"transcript_analysis_{timestamp}.md")

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Transcripts analyzed: {len(analyses)}")
    print(f"Report saved to: {output_path}")
    if args.save_json:
        print(f"JSON analyses saved to: {json_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
