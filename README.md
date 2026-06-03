# Convo Dataset Pipeline

Synthetic dataset pipeline for full-duplex speech LLM training.

The v1 pipeline focuses on:

- Stage 3: clean two-speaker English dialogue generated from scripts and synthesized with Dia 1.6B.
- Stage 4: overlapped dialogue derived from Stage 3 with heuristic backchannels, interruptions, and simultaneous starts.
- Mimi token exports using the first 8 codebooks for training shards.
- Hugging Face dataset upload by subset.

## CLI

```bash
convo-ds --help
```

Main commands:

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
export LLM_API_KEY=...
export LLM_BASE_URL=https://integrate.api.nvidia.com/v1
export LLM_MODEL=...
```

Hugging Face upload uses:

```bash
export HF_TOKEN=...
export HF_DATASET_REPO=org-or-user/dataset-name
```

GPU synthesis/tokenization requires optional dependencies and model access.

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
