# SpeechAI

Real-time speech analysis for CEAT tyre sales assistants. Transcribes live audio, analyzes customer sentiment and persona, detects competitor mentions, and provides actionable coaching suggestions in real-time.

## Features

- **Real-time transcription** with Azure Speech or Gemini 2.0 Flash
- **Multi-agent analysis** running in parallel for low latency:
  - Sentiment analysis with confidence scores and signal detection
  - Persona inference (6 CEAT customer personas)
  - Product analysis with upsell opportunity detection
  - Competitor intelligence with counter-positioning
  - Contextual sales scripts and objection handlers
- **Actionable suggestions** consolidated from all agents (2-3 bullets max)
- **Conversation context** tracking throughout the session
- **Per-agent model configuration** for optimal performance

## Architecture

```
Microphone → Transcription → Parallel Agents ──────────────────→ Consolidator → Display
                                    │                                   │
                                    ├── Sentiment Agent (gpt-5-mini)    │
                                    ├── Persona Agent (gpt-5-mini)      │
                                    ├── Product Agent (gpt-5-mini)      ├→ 2-3 Suggestions
                                    ├── Competition Agent (gpt-5-mini)  │
                                    └── Sales Prompts Agent (haiku)     │
                                                                        ↓
                                                            claude-haiku-4-5
```

### Agent Overview

| Agent | Purpose | Default Model |
|-------|---------|---------------|
| **Sentiment** | Detects positive/negative/neutral sentiment with signals | gpt-5-mini |
| **Persona** | Infers customer persona from 6 CEAT segments | gpt-5-mini |
| **Product** | Identifies products mentioned, upsell opportunities | gpt-5-mini |
| **Competition** | Detects competitor mentions, provides counter-positioning | gpt-5-mini |
| **Sales Prompts** | Retrieves contextual scripts and objection handlers | claude-haiku-4-5 |
| **Consolidator** | Combines all outputs into 2-3 actionable suggestions | claude-haiku-4-5 |

### CEAT Domain Knowledge

**6 Customer Personas:**
- Entitled Evan (premium demanding)
- Impatient Ashish (time-sensitive)
- Pragmatic Purnima (safety/hassle-free)
- Thorough Tushar (research-oriented)
- Savvy Sarabh (value-seeking)
- Bindaas Bharat (durability-focused)

**8 CEAT Products:** SportDrive SUV CALM, SportDrive, CrossDrive AT, SecuraDrive SUV, SecuraDrive, Energy Drive, Milaze X5, Milaze X3

**6 Competitors Tracked:** MRF, Apollo, JK Tyre, Bridgestone, Michelin, Goodyear

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone the repository
git clone https://github.com/yourusername/speechAI.git
cd speechAI

# Install dependencies
uv sync
```

### System Dependencies

- **ffmpeg** - Required for audio file processing (MP3 conversion)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Python Dependencies

Key libraries (installed automatically via `uv sync`):
- **textual** - Terminal UI framework for the dashboard
- **azure-cognitiveservices-speech** - Azure Speech SDK
- **litellm** - LLM routing for agents
- **sounddevice** / **webrtcvad** - Audio capture and voice activity detection

## Configuration

Create a `.env` file with your credentials:

```bash
# Azure Speech (for Azure mode)
AZURE_SPEECH_KEY=your_key
AZURE_SPEECH_ENDPOINT=your_endpoint

# LiteLLM proxy (agents use this)
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=sk-1234

# Gemini transcription (for Gemini mode)
GEMINI_MODEL=vertex_ai/gemini-2.0-flash
```

### Per-Agent Model Configuration

Each agent has a default model optimized for its task. Override via environment variables:

```bash
# JSON output agents (reliable structured output)
LITELLM_MODEL_SENTIMENT=gpt-5-mini
LITELLM_MODEL_PERSONA=gpt-5-mini
LITELLM_MODEL_PRODUCT=gpt-5-mini
LITELLM_MODEL_COMPETITION=gpt-5-mini

# Tone-sensitive agents (customer-facing scripts)
LITELLM_MODEL_SALES_PROMPTS=claude-haiku-4.5
LITELLM_MODEL_CONSOLIDATOR=claude-haiku-4.5
```

**Model recommendations:**
| Model | Best For |
|-------|----------|
| `gpt-5-mini` | Reliable JSON output, structured data extraction |
| `gemini-2.5-flash-lite` | Large context, cost-effective |
| `deepseek-v3.1` | Strong reasoning, cost-effective |
| `claude-haiku-4-5` | Natural conversational tone, customer-facing text |

## Usage

### Live Microphone Mode (Console)

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

### UI Mode (Textual Dashboard)

Real-time dashboard with fixed panels that update as utterances are detected:

```bash
# Live microphone with Gemini (default)
uv run speechai-ui

# Live microphone with Azure
uv run speechai-ui --backend azure

# Stream from audio file
uv run speechai-ui --file recording.mp3

# Stream from file with Azure backend
uv run speechai-ui --file recording.mp3 --backend azure
```

**UI Layout:**
- Fixed panels for Sentiment, Persona, Product, Competitors
- Suggestions panel with 2-3 actionable bullets
- Real-time latency display
- Scrollable history log
- Keyboard shortcuts: `r`=reset, `m`=mute, `q`=quit

### File Streaming Mode (Console)

Stream audio files through the pipeline with real-time simulation:

```bash
# Stream file with Gemini (default, real-time pacing)
uv run speechai-file recording.mp3

# Stream file with Azure
uv run speechai-file recording.mp3 --backend azure

# Fast mode (no real-time pacing)
uv run speechai-file recording.mp3 --no-realtime

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
# Full analysis: analyze transcripts and generate report
uv run python scripts/analyze_transcripts.py

# Report only: regenerate report from existing analysis files
uv run python scripts/analyze_transcripts.py --report-only
```

**Output:**
- `data/analysis/*_analysis.json` - Individual analysis per call
- `data/analysis/report_*.md` - Combined executive report (for agents & dealers)

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
| `--report-only` | Generate report only from existing analysis JSON files |

#### What Analysis Extracts

**Per call:** summary, outcome, sentiment, objections, customer pain points, sales rep performance, action items, risk flags

**Combined report:** executive summary, outcomes breakdown, customer insights, objection patterns, recommendations

## Output Example

```
[14:23:45] Customer │ INTERESTED (78%) │ Pragmatic Purnima
  "I'm looking for tyres for my Innova. Safety is important, we do a lot of highway driving.
   But I've heard MRF is more durable..."

  Suggestions:
    → Recommend SecuraDrive SUV - excellent wet grip for highway safety
    → Counter MRF: "CEAT offers similar durability with CALM noise reduction technology"
    → Mention run-flat capability for highway peace of mind

  Persona: Pragmatic Purnima (safety-focused, hassle-free)
  Products: SecuraDrive SUV, Run-flat option
  Competitor: MRF mentioned
  Upsell: Run-flat tyres (highway safety trigger)

  [Azure STT: 1250ms | 5 Agents: 487ms | Consolidator: 312ms | Total: 2049ms]
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
  Personas detected: Pragmatic Purnima (7x), Thorough Tushar (3x)
  Products discussed: SecuraDrive SUV, SportDrive
  Competitors mentioned: MRF (2x), Apollo (1x)
  Upsell opportunities: run_flat, calm_technology
────────────────────────────────────
```

## Project Structure

```
├── data/
│   ├── phrases.txt          # Phrase list for recognition (70+ CEAT terms)
│   ├── transcripts/         # JSON files from Azure Speech
│   ├── translations/        # Formatted English transcripts
│   └── analysis/            # Analysis JSON + reports
│
├── scripts/
│   ├── batch_transcribe.py  # Batch transcription + translation
│   └── analyze_transcripts.py # LLM-based transcript analysis
│
└── src/speechai/
    ├── main.py              # Azure mode entry point (console)
    ├── main_gemini.py       # Gemini mode entry point (console)
    ├── main_file.py         # File streaming entry point (console)
    ├── main_ui.py           # Textual UI entry point (dashboard)
    ├── ui.py                # Textual UI components
    ├── assistant_base.py    # Shared assistant logic + orchestrator
    ├── display.py           # Terminal output formatting
    ├── context.py           # Conversation context tracking
    ├── transcription.py     # Azure Speech transcriber
    ├── transcription_gemini.py  # Gemini transcriber
    ├── transcription_file.py    # File-based transcription utilities
    ├── prompts.yaml         # Agent prompts + CEAT domain knowledge
    └── agents/
        ├── __init__.py      # Exports all agents and constants
        ├── base.py          # BaseAgent with per-agent model config
        ├── sentiment.py     # Sentiment + signal detection
        ├── persona.py       # 6 CEAT personas with triggers
        ├── product.py       # CEAT products + upsell toolkit
        ├── competition.py   # Competitor intelligence database
        ├── sales_prompts.py # Upsell scripts + objection handlers
        └── consolidator.py  # Consolidator + AgentOrchestrator
```

## Extending

### Adding New Agents

1. Create a new agent in `src/speechai/agents/`:
   ```python
   from speechai.agents.base import AgentResult, BaseAgent

   class MyAgent(BaseAgent):
       name = "my_agent"
       default_model = "gpt-5-mini"  # or claude-haiku-4-5 for tone

       async def analyze(self, text: str, context: dict | None = None) -> AgentResult:
           # Your analysis logic
           return AgentResult(agent_name=self.name, success=True, data={...}, latency_ms=0)
   ```
2. Add to `AgentOrchestrator.initialize()` in `consolidator.py`
3. Include in `asyncio.gather()` call in `AgentOrchestrator.process()`
4. Update `Consolidator.analyze()` to use new agent's output
5. Add prompts to `prompts.yaml` under your agent's name

### Customizing Prompts

Edit `src/speechai/prompts.yaml` to customize:
- Sentiment detection criteria and signal keywords
- Persona triggers and characteristics
- Product catalog and upsell triggers
- Competitor counter-positioning
- Sales scripts and objection handlers
- Consolidator suggestion generation rules

### Embedded Domain Knowledge

Each agent embeds CEAT-specific knowledge as fallback:
- `persona.py`: `PERSONAS` dict with 6 customer segments
- `product.py`: `CEAT_PRODUCTS` and `UPSELL_TOOLKIT`
- `competition.py`: `COMPETITORS` and `CEAT_DIFFERENTIATORS`
- `sales_prompts.py`: `UPSELL_SCRIPTS` and `OBJECTION_HANDLERS`

This ensures agents can detect triggers even if LLM parsing fails.

## License

MIT License - see [LICENSE](LICENSE) for details.
