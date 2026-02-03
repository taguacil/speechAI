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

### Batch Transcription & Analysis Pipeline

For batch processing multiple audio files with Azure Speech, translation, and LLM analysis.

**Directory structure:**

```
data/
├── phrases.txt       # Phrase list for improved recognition
├── transcripts/      # JSON files from Azure Speech
├── translations/     # Formatted English transcripts (_en.txt)
└── analysis/         # Analysis JSON files + combined report
```

**Environment variables:**

```bash
# Azure Speech & Storage (required for transcription)
AZURE_SPEECH_KEY=your_key
AZURE_SPEECH_REGION=swedencentral
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...

# For Azure translation:
AZURE_TRANSLATOR_KEY=your_key
AZURE_TRANSLATOR_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com

# For LLM translation (--llm flag):
LLM_BASE_URL=http://localhost:4000
LLM_API_KEY=sk-1234
LLM_MODEL=kimi-2.5
```

#### Transcribe

```bash
# Transcribe audio files (auto-detect language)
uv run python scripts/batch_transcribe.py ./data/recordings
```

#### Translate

Converts JSON transcripts to formatted English with timestamps and speaker labels.

```bash
# Using Azure Translator
uv run python scripts/batch_transcribe.py --translate-only

# Using LLM (recommended for better quality)
uv run python scripts/batch_transcribe.py --translate-only --llm
```

**Output format (`data/translations/*_en.txt`):**

```
[0.0s] Customer: Hello, I'm calling about your service
[3.5s] Sales Rep: Welcome! How can I help you?
[8.2s] Customer: I want to know the pricing details
         (Original hi-IN: मुझे कीमत की जानकारी चाहिए)
```

#### Analyze

```bash
uv run python scripts/analyze_transcripts.py
```

**Output:**
- `data/analysis/*_analysis.json` - Individual analysis per call
- `data/analysis/report_*.md` - Combined executive report

#### Combined Commands

```bash
# Transcribe + translate in one step
uv run python scripts/batch_transcribe.py ./data/recordings --translate --llm

# Download from existing job + translate
uv run python scripts/batch_transcribe.py --job-id <id> --translate --llm

# Use existing Azure container
uv run python scripts/batch_transcribe.py --container <name> --translate --llm
```

#### Phrase List

Create `data/phrases.txt` to improve recognition of domain-specific terms:

```
# One phrase per line, comments start with #
product name
company name
technical term
```

#### Options Reference

**batch_transcribe.py:**

| Option | Description |
|--------|-------------|
| `--translate` | Translate after transcription |
| `--translate-only` | Only translate existing JSON files |
| `--llm` | Use LLM for translation (instead of Azure) |
| `--job-id` | Download from existing transcription job |
| `--container` | Use existing Azure container (skip upload) |
| `--phrases` | Phrase list file (default: ./data/phrases.txt) |
| `--locale` | Language locale (auto-detect if not set) |

**analyze_transcripts.py:**

| Option | Description |
|--------|-------------|
| `--input` | Input directory (default: ./data/translations) |
| `--output` | Output directory (default: ./data/analysis) |
| `--model` | LLM model (default: gemini-2.5-pro) |

#### What Analysis Extracts

**Per call:** summary, outcome, sentiment, objections, customer pain points, sales rep performance, action items, risk flags

**Combined report:** executive summary, outcomes breakdown, customer insights, objection patterns, recommendations

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
├── data/
│   ├── phrases.txt          # Phrase list for recognition
│   ├── transcripts/         # JSON files from Azure Speech
│   ├── translations/        # Formatted English transcripts
│   └── analysis/            # Analysis JSON + reports
│
├── scripts/
│   ├── batch_transcribe.py  # Batch transcription + translation
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
