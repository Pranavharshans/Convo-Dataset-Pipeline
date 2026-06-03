---
license: apache-2.0
language:
  - en
task_categories:
  - text-to-speech
  - automatic-speech-recognition
  - conversational
pretty_name: Convo Dataset Pipeline Output
---

# Dataset

Synthetic English two-speaker dialogue data for full-duplex speech LLM experiments.

## Subsets

- `scripts`: text-only dialogue scripts generated from an OpenAI-compatible LLM API.
- `stage3`: clean alternating two-speaker dialogue audio and metadata.
- `stage4`: overlapped dialogue audio and metadata derived from Stage 3.
- `shards`: ready-to-train JSONL sequences with speaker markers, text tokens, Mimi tokens, and loss masks.

## Generation

- Dia model: `nari-labs/Dia-1.6B`
- Mimi model: `kyutai/mimi`
- Training codebooks: first 8 Mimi codebooks
- Audio format: WAV

## Safety

This dataset is synthetic and should not be used to imitate real people or produce deceptive audio.
