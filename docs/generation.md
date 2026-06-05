# Generation Runbook

This runbook covers the text-script generation path used before ZipVoice-Dialog speech synthesis.

The main command is:

```bash
.venv/bin/convo-ds generate-zipvoice-dialog-scripts \
  --config local.mistral.toml \
  --output-dir data/scripts/zipvoice_dialog \
  --limit 50000
```

The command is resumable. If `zipvoice_dialog_scripts.jsonl` already exists, it counts accepted conversations and continues from the next missing ID. Rejections are expected; they mean the validator discarded outputs with the wrong turn count, word count, or stage directions.

## Install

```bash
cd ~/Convo-Dataset-Pipeline
uv sync --extra dev
```

If `convo-ds` is not on `PATH`, use `.venv/bin/convo-ds`.

## Mistral Provider

Start from the checked-in Mistral example:

```bash
cp examples/mistral.config.toml local.mistral.toml
export MISTRAL_API_KEY="replace-with-your-key"
```

Recommended high-throughput settings for `mistral-small-2506`:

```toml
base_url = "https://api.mistral.ai/v1"
api_key_env = "MISTRAL_API_KEY"
model = "mistral-small-2506"
requests_per_minute = 180
conversations_per_request = 10
concurrency = 8
```

Run generation:

```bash
.venv/bin/convo-ds generate-zipvoice-dialog-scripts \
  --config local.mistral.toml \
  --output-dir data/scripts/zipvoice_dialog \
  --limit 50000
```

## NVIDIA NIM Provider

Start from the checked-in NIM example:

```bash
cp examples/nim.config.toml local.nim.toml
export NVIDIA_API_KEY="replace-with-your-key"
```

The default NIM config uses:

```toml
base_url = "https://integrate.api.nvidia.com/v1"
api_key_env = "NVIDIA_API_KEY"
model = "openai/gpt-oss-20b"
requests_per_minute = 35
conversations_per_request = 5
concurrency = 1
```

Run generation:

```bash
.venv/bin/convo-ds generate-zipvoice-dialog-scripts \
  --config local.nim.toml \
  --output-dir data/scripts/zipvoice_dialog \
  --limit 50000
```

Do not commit `local.*.toml` files or API keys. If a key was pasted into terminal history or chat, rotate it.

## Output Layout

Generation writes:

```text
data/scripts/zipvoice_dialog/
  zipvoice_dialog_scripts.jsonl
  short.jsonl
  medium.jsonl
  long.jsonl
  extended.jsonl
  zipvoice_dialog_scripts.manifest.json
```

Expected final counts:

```text
zipvoice_dialog_scripts.jsonl  50000
short.jsonl                    15000
medium.jsonl                   20000
long.jsonl                     10000
extended.jsonl                  5000
```

Check progress:

```bash
wc -l data/scripts/zipvoice_dialog/zipvoice_dialog_scripts.jsonl
wc -l data/scripts/zipvoice_dialog/short.jsonl
wc -l data/scripts/zipvoice_dialog/medium.jsonl
wc -l data/scripts/zipvoice_dialog/long.jsonl
wc -l data/scripts/zipvoice_dialog/extended.jsonl
```

## Reset And Resume

Resume:

```bash
.venv/bin/convo-ds generate-zipvoice-dialog-scripts \
  --config local.mistral.toml \
  --output-dir data/scripts/zipvoice_dialog \
  --limit 50000
```

Start from scratch:

```bash
mv data/scripts/zipvoice_dialog data/scripts/zipvoice_dialog.old
.venv/bin/convo-ds generate-zipvoice-dialog-scripts \
  --config local.mistral.toml \
  --output-dir data/scripts/zipvoice_dialog \
  --limit 50000
```

If `git pull` is blocked by a local edit to `examples/config.toml`, keep your provider config in `local.mistral.toml` or `local.nim.toml`, then restore the example file:

```bash
git restore examples/config.toml
git pull
```

If Git says the file is unmerged, inspect with `git status --short` before continuing.

## Validate Scripts

```bash
.venv/bin/convo-ds validate \
  --subset scripts \
  --path data/scripts/zipvoice_dialog/zipvoice_dialog_scripts.jsonl
```

## Export ZipVoice TSV

Create a prompt bank:

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

Export:

```bash
.venv/bin/convo-ds export-zipvoice-dialog-tsv \
  --scripts data/scripts/zipvoice_dialog/zipvoice_dialog_scripts.jsonl \
  --voice-prompts voice_prompts \
  --output data/zipvoice_dialog/test.tsv
```

Run ZipVoice:

```bash
python3 -m zipvoice.bin.infer_zipvoice_dialog \
  --model-name zipvoice_dialog_stereo \
  --test-list data/zipvoice_dialog/test.tsv \
  --res-dir data/zipvoice_dialog/results
```

## Hugging Face Upload

Set credentials:

```bash
export HF_TOKEN="replace-with-your-token"
export HF_DATASET_REPO="username-or-org/dataset-name"
```

Dry run first:

```bash
.venv/bin/convo-ds upload-hf \
  --config local.mistral.toml \
  --data-dir data \
  --subset scripts \
  --dry-run
```

Upload scripts:

```bash
.venv/bin/convo-ds upload-hf \
  --config local.mistral.toml \
  --data-dir data \
  --subset scripts
```

Use `--subset all` only after audio, Stage 4, Mimi tokens, and shards are complete.
