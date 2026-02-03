# SpeechAI

Real-time speech sentiment analysis for sales assistants. Transcribes live audio, analyzes sentiment, and provides actionable coaching suggestions in real-time.

## Features

- **Real-time transcription** with Azure Speech or Gemini 2.0 Flash
- **Sentiment analysis** (positive/negative/neutral) with confidence scores
- **Signal detection** (objections, interest, budget concerns, etc.)
- **Actionable suggestions** for sales reps in real-time
- **Conversation context** tracking throughout the session
- **Session management** with reset and mute controls

## Architecture

```
Microphone → Transcription → Parallel Agents → Consolidator → Display
                                    │
                                    ├── Sentiment Agent
                                    └── (extensible for more agents)
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone https://github.com/yourusername/speechAI.git
cd speechAI

# Install dependencies
uv sync
```

### Dependencies

- **ffmpeg** - Required for audio file processing (MP3 conversion)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## Configuration

Create a `.env` file with your credentials:

```bash
# Azure Speech (for Azure mode)
AZURE_SPEECH_KEY=your_key
AZURE_SPEECH_ENDPOINT=your_endpoint

# LiteLLM proxy (for Gemini mode)
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=sk-1234
GEMINI_MODEL=gemini-2.0-flash-vertex

# LLM for agents (sentiment, consolidator)
LLM_BASE_URL=http://localhost:4000
LLM_API_KEY=sk-1234
LLM_MODEL=your-model
```

## Usage

### Live Microphone Mode

```bash
# Azure Speech transcription
uv run speechai

# Gemini transcription
uv run speechai-gemini
```

**Keyboard commands during session:**
- `r` - Reset session (clear context, start fresh)
- `m` - Mute/unmute (pause processing)
- `q` - Quit

### File Processing Mode

Process recorded audio files for testing:

```bash
# Single file (Gemini backend)
uv run speechai-file recording.mp3

# Single file (Azure backend)
uv run speechai-file recording.mp3 --backend azure

# Process all files in directory
uv run speechai-file recordings/
```

Supported formats: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`

### Batch Transcription

For batch processing multiple audio files with Azure Speech (higher quality, speaker diarization, translation).

**Additional environment variables:**

```bash
# Azure Storage (for uploading audio files)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...

# Azure Speech region
AZURE_SPEECH_REGION=swedencentral

# Azure Translator (for translation)
AZURE_TRANSLATOR_KEY=your_key
AZURE_TRANSLATOR_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com
```

**Basic usage - upload and transcribe:**

```bash
# Transcribe all files in a directory (auto-detect language)
uv run python scripts/batch_transcribe.py ./data/recordings --output ./data/transcripts

# With specific language
uv run python scripts/batch_transcribe.py ./data/recordings --locale es-ES --output ./data/transcripts
```

**With translation to English:**

```bash
# Transcribe and translate (auto-detect source language)
uv run python scripts/batch_transcribe.py ./data/recordings --translate --output ./data/transcripts

# Specify source language for translation
uv run python scripts/batch_transcribe.py ./data/recordings --translate --source-lang es
```

**Use existing Azure container (skip upload):**

```bash
# If files are already uploaded to Azure Blob Storage
uv run python scripts/batch_transcribe.py --container my-container-name --translate
```

**Download from existing transcription job:**

```bash
# Just download results from a completed job
uv run python scripts/batch_transcribe.py --job-id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --output ./data/transcripts

# Download and translate
uv run python scripts/batch_transcribe.py --job-id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --translate
```

**Translate only (existing .txt files):**

```bash
# Translate already transcribed files
uv run python scripts/batch_transcribe.py --translate-only --output ./data/transcripts

# Specify source language
uv run python scripts/batch_transcribe.py --translate-only --source-lang es --output ./data/transcripts
```

**Output files per recording:**

```
data/transcripts/
├── recording-001.json      # Full Azure response (timing, confidence, words)
├── recording-001.txt       # Plain text transcription
├── recording-001_en.txt    # English translation (if --translate)
└── ...
```

**All options:**

| Option | Description |
|--------|-------------|
| `--output` | Output directory (default: ./transcripts) |
| `--locale` | Language locale, e.g., es-ES (auto-detect if not set) |
| `--container` | Use existing Azure container name (skip upload) |
| `--job-id` | Download from existing transcription job |
| `--translate` | Translate transcripts to English |
| `--translate-only` | Only translate existing .txt files |
| `--source-lang` | Source language for translation (auto-detect if not set) |
| `--target-lang` | Target language (default: en) |
| `--diarization` | Enable speaker diarization |
| `--min-speakers` | Min speakers for diarization (default: 1) |
| `--max-speakers` | Max speakers for diarization (default: 5) |
| `--timeout` | Timeout in seconds (default: 3600) |

### Transcript Analysis (LLM)

Analyze transcripts using Claude Opus or Gemini Pro to extract insights and generate comprehensive reports.

```bash
# Analyze all transcripts in a directory
uv run python scripts/analyze_transcripts.py ./data/transcripts

# Use specific model
uv run python scripts/analyze_transcripts.py ./data/transcripts --model gemini-3-pro

# Save report to specific file
uv run python scripts/analyze_transcripts.py ./data/transcripts --output ./reports/analysis.md

# Also save individual analyses as JSON
uv run python scripts/analyze_transcripts.py ./data/transcripts --json
```

**What it extracts per call:**
- Call summary and outcome
- Caller type and intent
- Sentiment analysis with progression
- Products/services mentioned
- Objections raised and how they were handled
- Customer pain points and priorities
- Sales rep performance (strengths, areas for improvement)
- Action items and follow-up notes
- Risk flags

**Combined report includes:**
- Executive summary
- Call outcomes breakdown
- Customer insights across all calls
- Objection patterns and handling
- Sales team performance analysis
- Recommendations

## Output Example

```
[14:23:45] Speaker-1 │ NEGATIVE (85%)
  "That's more than we budgeted for this quarter"
  Signals: price objection
  Suggestions:
    → Acknowledge budget concern directly
    → Ask about their timeline flexibility
  [Azure STT: 1250ms | Agents: 487ms | Total: 1737ms]
```

## Session Summary

At the end of each session (or on reset), you'll see:

```
────────────────────────────────────
Session Summary:
  Utterances: 12
  Duration: 145s
  Sentiment: {'positive': 4, 'negative': 3, 'neutral': 5}
  Signals: {'budget': 2, 'interest': 3, 'objection': 1}
────────────────────────────────────
```

## Project Structure

```
├── scripts/
│   ├── batch_transcribe.py   # Batch transcription + translation
│   └── analyze_transcripts.py # LLM-based transcript analysis
│
└── src/speechai/
    ├── main.py              # Azure mode entry point
    ├── main_gemini.py       # Gemini mode entry point
    ├── main_file.py         # File processing entry point
    ├── assistant_base.py    # Shared assistant logic
    ├── display.py           # Terminal output formatting
    ├── context.py           # Conversation context tracking
    ├── transcription.py     # Azure Speech transcriber
    ├── transcription_gemini.py  # Gemini transcriber
    ├── transcription_file.py    # File-based transcription
    ├── prompts.yaml         # Agent prompts configuration
    └── agents/
        ├── base.py          # Base agent class
        ├── sentiment.py     # Sentiment analysis agent
        └── consolidator.py  # Suggestion consolidator
```

## Extending

### Adding New Agents

1. Create a new agent in `src/speechai/agents/`
2. Inherit from `BaseAgent`
3. Add to the orchestrator in `consolidator.py`

### Customizing Prompts

Edit `src/speechai/prompts.yaml` to customize:
- Sentiment detection criteria
- Suggestion generation rules
- Signal keywords

## License

MIT License - see [LICENSE](LICENSE) for details.
