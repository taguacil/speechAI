#!/usr/bin/env python
"""LLM-based analysis of transcripts.

Reads translated English transcripts (_en.txt) from data/translations.
Outputs individual _analysis.json files and combined report to data/analysis.

Usage:
    # Analyze with defaults (data/translations -> data/analysis)
    python analyze_transcripts.py

    # Use specific model
    python analyze_transcripts.py --model gemini-2.5-pro

    # Custom directories
    python analyze_transcripts.py --input ./translations --output ./analysis

Environment variables:
    LLM_BASE_URL - LiteLLM proxy URL (default: http://localhost:4000)
    LLM_API_KEY - API key for LiteLLM
    LLM_MODEL - Default model to use
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import openai
from dotenv import load_dotenv

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


INDIVIDUAL_ANALYSIS_PROMPT = """You are a transcript expert analyzer. You will get outbound calls from current and potential customers at the call center of CEAT, a tyre company in India.
You job is to analyze the sales call transcript and extract comprehensive information according to the instructions below formalized in JSON format. The ultimate goal of the analysis is to provide insights, recommendations and a guide to improve the sales process.

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
    "key_themes": ["theme1", "theme2"],
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


def get_transcript_files(input_path: Path) -> list[Path]:
    """Get all translated English transcript files (_en.txt)."""
    if input_path.is_file() and input_path.name.endswith("_en.txt"):
        return [input_path]

    return sorted(input_path.glob("**/*_en.txt"))


def main():
    parser = argparse.ArgumentParser(description="Analyze transcripts with LLM")
    parser.add_argument("--input", type=Path, default=Path("./data/translations"), help="Input directory with _en.txt files")
    parser.add_argument("--output", type=Path, default=Path("./data/analysis"), help="Output directory for analysis files")
    parser.add_argument("--model", type=str, default=None, help="LLM model to use")

    args = parser.parse_args()
    load_dotenv()

    # Get LLM config
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:4000")
    api_key = os.getenv("LLM_API_KEY", "sk-1234")
    model = args.model or os.getenv("LLM_MODEL", "gemini-2.5-pro")

    logger.info(f"Using model: {model}")

    # Get transcript files
    files = get_transcript_files(args.input)
    if not files:
        logger.error("No translated transcript files (*_en.txt) found")
        sys.exit(1)

    logger.info(f"Found {len(files)} transcript files")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = TranscriptAnalyzer(base_url, api_key, model)

    # Process each file
    analyses = []
    for file_path in files:
        logger.info(f"Processing: {file_path.name}")

        try:
            conversation = file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            continue

        if not conversation.strip():
            logger.warning(f"Empty transcript: {file_path.name}")
            continue

        # Analyze transcript
        analysis = analyzer.analyze_single(conversation, file_path.stem)
        analyses.append(analysis)

        # Save individual analysis JSON to output directory
        base_name = file_path.stem.replace("_en", "")
        analysis_path = args.output / f"{base_name}_analysis.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {analysis_path.name}")

    logger.info(f"Analyzed {len(analyses)} transcripts")

    if not analyses:
        logger.error("No transcripts were analyzed successfully")
        sys.exit(1)

    # Generate combined report
    report = analyzer.generate_combined_report(analyses)

    # Save report to output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = args.output / f"report_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Transcripts analyzed: {len(analyses)}")
    print(f"Individual analyses: {args.output}/*_analysis.json")
    print(f"Combined report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
