# Convo Dataset Pipeline

Synthetic dataset pipeline for full-duplex speech LLM training.

The v1 pipeline focuses on:

- ZipVoice-Dialog script generation with 50K bucketed conversations.
- Stage 3: clean two-speaker English dialogue generated from scripts and synthesized with ZipVoice or Dia adapters.
- Stage 4: overlapped dialogue derived from Stage 3 with heuristic backchannels, interruptions, and simultaneous starts.
- Mimi token exports using the first 8 codebooks for training shards.
- Hugging Face dataset upload by subset.

## CLI

```bash
convo-ds --help
```

Main commands:

- `generate-zipvoice-dialog-scripts`
- `export-zipvoice-dialog-tsv`
- `generate-scripts`
- `synth-stage3`
- `inject-overlap`
- `tokenize-mimi`
- `assemble-shards`
- `validate`
- `upload-hf`

## Environment

Script generation uses an OpenAI-compatible API:

```bash
export MISTRAL_API_KEY=...
# or
export NVIDIA_API_KEY=...
```

Use provider-specific local configs instead of editing checked-in examples:

```bash
cp examples/mistral.config.toml local.mistral.toml
cp examples/nim.config.toml local.nim.toml
```

Full text generation runbook: [docs/generation.md](docs/generation.md).

Real-time Qwen + Mimi training plan: [docs/realtime_training_plan.md](docs/realtime_training_plan.md).

Hugging Face upload uses:

```bash
export HF_TOKEN=...
export HF_DATASET_REPO=org-or-user/dataset-name
```

GPU synthesis/tokenization requires optional dependencies and model access.

## ZipVoice Dialogue Scripts

The primary script path is ZipVoice-Dialog-Stereo. Generate bucketed scripts:

```bash
convo-ds generate-zipvoice-dialog-scripts \
  --config local.mistral.toml \
  --output-dir data/scripts/zipvoice_dialog \
  --limit 50000
```

This writes:

```text
data/scripts/zipvoice_dialog/
  zipvoice_dialog_scripts.jsonl
  short.jsonl
  medium.jsonl
  long.jsonl
  extended.jsonl
  zipvoice_dialog_scripts.manifest.json
```

The generator enforces separate bucket constraints:

- short: 4-6 turns, 28-55 words, fast exchanges.
- medium: 6-8 turns, 70-115 words, natural flow.
- long: 8-10 turns, 115-170 words, sustained topic.
- extended: 10-14 turns, 170-260 words, story arc or deeper explanation.

During generation the terminal logs progress every `progress_every` accepted conversations. The default is 500:

```text
[zipvoice-scripts] completed=500/50000 bucket=short bucket_completed=500
```

For high-throughput OpenAI-compatible providers, use controlled concurrency. For Mistral Small 2506, a conservative starting point is:

```toml
requests_per_minute = 180
conversations_per_request = 10
concurrency = 8
```

For NVIDIA NIM, start with `examples/nim.config.toml` and tune upward only after the provider is stable.

Export ZipVoice split-prompt TSV:

```bash
convo-ds export-zipvoice-dialog-tsv \
  --scripts data/scripts/zipvoice_dialog/zipvoice_dialog_scripts.jsonl \
  --voice-prompts voice_prompts \
  --output data/zipvoice_dialog/test.tsv
```

Expected prompt bank format:

```text
voice_prompts/
  pair_000/
    s1.wav
    s2.wav
    meta.json
```

`meta.json`:

```json
{
  "s1_prompt_transcription": "Hello, this is speaker one.",
  "s2_prompt_transcription": "Hi, this is speaker two.",
  "s1_prompt_wav": "s1.wav",
  "s2_prompt_wav": "s2.wav"
}
```

Run ZipVoice:

```bash
python3 -m zipvoice.bin.infer_zipvoice_dialog \
  --model-name zipvoice_dialog_stereo \
  --test-list data/zipvoice_dialog/test.tsv \
  --res-dir data/zipvoice_dialog/results
```

## Local Smoke Test

The local smoke path uses mock Dia and mock Mimi adapters, so it does not need a GPU:

```bash
uv sync --extra dev

convo-ds synth-stage3 \
  --mock \
  --scripts examples/fixtures/scripts.jsonl \
  --output /private/tmp/convo_stage3 \
  --limit 1

convo-ds inject-overlap \
  --stage3 /private/tmp/convo_stage3 \
  --output /private/tmp/convo_stage4 \
  --limit 1

convo-ds tokenize-mimi \
  --mock \
  --stage /private/tmp/convo_stage3 \
  --output /private/tmp/convo_mimi.jsonl \
  --limit 1

convo-ds assemble-shards \
  --tokens /private/tmp/convo_mimi.jsonl \
  --output /private/tmp/convo_train.jsonl

convo-ds validate --subset stage3 --path /private/tmp/convo_stage3
convo-ds validate --subset stage4 --path /private/tmp/convo_stage4
convo-ds validate --subset shards --path /private/tmp/convo_train.jsonl
```

## Production Shape

Default config targets 50K Stage 3 conversations:

- 15K short, 4-6 turns, ~16s average.
- 20K medium, 6-8 turns, ~30s average.
- 10K long, 8-10 turns, ~45s average.
- 5K extended, 10-14 turns, ~65s average.

Stage 4 derives ~200h of overlapped dialogue from Stage 3 with a natural overlap mix.

See [GPU smoke test](docs/gpu_smoke_test.md) for the CUDA runbook.
