# GPU Smoke Test

Run this on a CUDA machine after installing Dia and the optional GPU dependencies.

```bash
uv sync --extra dev --extra gpu
```

Generate or reuse scripts:

```bash
convo-ds generate-scripts \
  --config examples/config.toml \
  --output data/scripts/dialogues.jsonl \
  --limit 10
```

Synthesize Stage 3 with Dia 1.6B:

```bash
convo-ds synth-stage3 \
  --config examples/config.toml \
  --scripts data/scripts/dialogues.jsonl \
  --output data/stage3 \
  --limit 3
```

Inject Stage 4 overlaps:

```bash
convo-ds inject-overlap \
  --config examples/config.toml \
  --stage3 data/stage3 \
  --output data/stage4 \
  --limit 3
```

Tokenize and assemble shards:

```bash
convo-ds tokenize-mimi \
  --config examples/config.toml \
  --stage data/stage3 \
  --output data/tokens/stage3_mimi.jsonl \
  --limit 3

convo-ds assemble-shards \
  --tokens data/tokens/stage3_mimi.jsonl \
  --output data/shards/stage3_train.jsonl
```

Validate before upload:

```bash
convo-ds validate --subset stage3 --path data/stage3
convo-ds validate --subset stage4 --path data/stage4
convo-ds validate --subset shards --path data/shards/stage3_train.jsonl
```
