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
