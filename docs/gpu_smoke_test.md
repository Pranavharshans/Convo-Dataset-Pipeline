# GPU Smoke Test

Run this on a CUDA machine after installing ZipVoice/Dia and the optional GPU dependencies.

```bash
uv sync --extra dev --extra gpu
```

Generate or reuse ZipVoice scripts. For provider setup, see [generation.md](generation.md).

```bash
convo-ds generate-zipvoice-dialog-scripts \
  --config local.mistral.toml \
  --output-dir data/scripts/zipvoice_dialog \
  --limit 10
```

Export ZipVoice TSV after creating a `voice_prompts/` prompt bank:

```bash
convo-ds export-zipvoice-dialog-tsv \
  --scripts data/scripts/zipvoice_dialog/zipvoice_dialog_scripts.jsonl \
  --voice-prompts voice_prompts \
  --output data/zipvoice_dialog/test.tsv
```

Synthesize with ZipVoice-Dialog-Stereo:

```bash
python3 -m zipvoice.bin.infer_zipvoice_dialog \
  --model-name zipvoice_dialog_stereo \
  --test-list data/zipvoice_dialog/test.tsv \
  --res-dir data/zipvoice_dialog/results
```

The legacy mock/Dia Stage 3 path is still useful for local end-to-end smoke checks:

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
