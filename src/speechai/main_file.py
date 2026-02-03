"""Process audio files for testing.

Run with:
    uv run speechai-file recording.mp3
    uv run speechai-file recording.mp3 --backend azure
    uv run speechai-file recordings/ --backend gemini
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from speechai.agents.consolidator import AgentOrchestrator
from speechai.assistant_base import load_prompts
from speechai.context import ConversationContext
from speechai.display import Colors, format_output


def get_audio_files(path: Path) -> list[Path]:
    """Get audio files from path (file or directory)."""
    audio_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    if path.is_file():
        return [path]
    elif path.is_dir():
        files = []
        for ext in audio_extensions:
            files.extend(path.glob(f"*{ext}"))
        return sorted(files)
    else:
        return []


async def process_file(
    audio_path: Path,
    transcriber,
    orchestrator: AgentOrchestrator,
    context: ConversationContext,
    backend: str,
) -> None:
    """Process a single audio file through the pipeline."""
    print(f"\n{Colors.CYAN}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}Processing: {audio_path.name}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")

    # Transcribe
    result = transcriber.transcribe(audio_path)

    if not result or not result.text:
        print(f"{Colors.RED}No transcription result{Colors.RESET}")
        return

    print(f"\n{Colors.DIM}Transcription ({result.latency_ms:.0f}ms):{Colors.RESET}")
    print(f"  \"{result.text}\"")

    # Get context and process through agents
    context_str = context.format_for_prompt(max_utterances=10)

    output = await orchestrator.process(
        text=result.text,
        speaker=result.speaker_id,
        conversation_context=context_str,
    )

    # Add to context
    context.add_utterance(
        text=result.text,
        speaker=result.speaker_id,
        sentiment=output.sentiment,
        confidence=output.confidence,
        signals=output.signals,
        # Multi-agent outputs
        persona_name=output.persona_name,
        products_mentioned=output.products_mentioned,
        competitors_mentioned=output.competitors_mentioned,
        upsell_opportunities=output.upsell_opportunities,
    )

    # Display results
    timestamp = datetime.now().strftime("%H:%M:%S")
    stt_label = "Gemini STT" if backend == "gemini" else "Azure STT"
    mode_color = Colors.MAGENTA if backend == "gemini" else Colors.BLUE

    print()
    format_output(
        timestamp=timestamp,
        speaker=result.speaker_id,
        text=result.text,
        sentiment=output.sentiment,
        confidence=output.confidence,
        signals=output.signals,
        suggestions=output.suggestions,
        stt_latency_ms=result.latency_ms,
        agents_latency_ms=output.total_latency_ms,
        mode_color=mode_color,
        stt_label=stt_label,
        # Multi-agent outputs
        persona_name=output.persona_name,
        persona_segment=output.persona_segment,
        products_mentioned=output.products_mentioned,
        upsell_opportunities=output.upsell_opportunities,
        recommended_product=output.recommended_product,
        competitors_mentioned=output.competitors_mentioned,
        counter_positioning=output.counter_positioning,
        objection_detected=output.objection_detected,
        upsell_script=output.upsell_script,
    )


def main() -> None:
    """Main entry point for file processing."""
    parser = argparse.ArgumentParser(
        description="Process audio files through the sales assistant pipeline."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Audio file or directory containing audio files",
    )
    parser.add_argument(
        "--backend",
        choices=["gemini", "azure"],
        default="gemini",
        help="Transcription backend (default: gemini)",
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"{Colors.RED}Path not found: {args.path}{Colors.RESET}")
        sys.exit(1)

    audio_files = get_audio_files(args.path)
    if not audio_files:
        print(f"{Colors.RED}No audio files found at: {args.path}{Colors.RESET}")
        sys.exit(1)

    load_dotenv()

    # Initialize transcriber
    if args.backend == "gemini":
        from speechai.transcription_file import FileTranscriberGemini
        transcriber = FileTranscriberGemini()
    else:
        from speechai.transcription_file import FileTranscriberAzure
        transcriber = FileTranscriberAzure()

    # Initialize orchestrator and context
    prompts = load_prompts()
    orchestrator = AgentOrchestrator(prompts)
    orchestrator.initialize()
    context = ConversationContext()

    # Print header
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}  Sales Assistant - File Processing Mode{Colors.RESET}")
    backend_color = Colors.MAGENTA if args.backend == "gemini" else Colors.BLUE
    print(f"{backend_color}  Backend: {args.backend.upper()}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.DIM}Files to process: {len(audio_files)}{Colors.RESET}")

    # Process files
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        for audio_file in audio_files:
            loop.run_until_complete(
                process_file(audio_file, transcriber, orchestrator, context, args.backend)
            )
    finally:
        loop.close()

    # Print session summary
    summary = context.get_conversation_summary()
    if summary["total_utterances"] > 0:
        print(f"\n{Colors.DIM}{'─' * 40}{Colors.RESET}")
        print(f"{Colors.BOLD}Session Summary:{Colors.RESET}")
        print(f"  Files processed: {len(audio_files)}")
        print(f"  Utterances: {summary['total_utterances']}")
        print(f"  Sentiment: {summary['sentiment_distribution']}")
        if summary['signal_counts']:
            print(f"  Signals: {summary['signal_counts']}")
        # Multi-agent tracking
        if summary.get('persona_counts'):
            persona_str = ", ".join(f"{k} ({v}x)" for k, v in summary['persona_counts'].items())
            print(f"  Personas: {persona_str}")
        if summary.get('products_discussed'):
            print(f"  Products: {', '.join(summary['products_discussed'][:5])}")
        if summary.get('competitor_counts'):
            comp_str = ", ".join(f"{k} ({v}x)" for k, v in summary['competitor_counts'].items())
            print(f"  Competitors: {comp_str}")
        if summary.get('upsell_opportunities'):
            print(f"  Upsells: {', '.join(summary['upsell_opportunities'][:3])}")
        print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")

    print(f"\n{Colors.DIM}Done.{Colors.RESET}")


if __name__ == "__main__":
    main()
