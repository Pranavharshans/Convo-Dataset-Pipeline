# Project 1: Streaming Mini-Miso TTS

## Goal

Build a small MisoTTS-style text-to-speech model before attempting any dialogue or duplex behavior.

The target model is:

```text
Qwen3 0.6B temporal backbone
+ 32-codebook Mimi depth decoder
+ frozen Mimi waveform decoder
```

This project proves that the model can generate intelligible speech tokens from text and can run in a streaming frame-by-frame loop. It is a single-stream TTS model, not a conversation model.

Non-goals:

```text
no user audio input
no user/assistant stream format
no conversation state
no turn-taking
no interruptions
no full-duplex behavior
```

Those are handled in later projects.

## Why This Comes First

Full duplex is too hard to debug if the base TTS model cannot already produce stable audio. This project isolates the speech generation problem:

```text
text -> Mimi codes -> waveform
```

Once this works, later projects can wrap the same audio backbone with dialogue timing. Project 1 itself stays text-to-speech only.

## Architecture

Use the Miso-style temporal/depth split:

```text
text tokens
previous audio frames
    ->
Qwen3 temporal backbone
    ->
cb0 for next Mimi frame
    ->
small depth decoder
    ->
cb1, cb2, ..., cb31
    ->
Mimi decoder
    ->
24 kHz waveform
```

The temporal backbone runs once per Mimi frame:

```text
12.5 frames/sec
80 ms per frame
```

The depth decoder predicts the within-frame codebook stack:

```text
cb1 conditioned on hidden + cb0
cb2 conditioned on hidden + cb0 + cb1
...
cb31 conditioned on hidden + cb0 ... cb30
```

This avoids flattening 32 codebooks into 400 large-model tokens/sec.

## Model Components

Recommended starting configuration:

```text
temporal backbone: Qwen/Qwen3-0.6B
audio codec: kyutai/mimi
codebooks: 32
codebook vocab size: about 2048
depth decoder size: 50M-150M parameters
training mode: LoRA on Qwen first, full or partial depth decoder training
```

Core modules:

```text
Qwen text embeddings
32 audio codebook embedding tables
summed audio-frame embedding
Qwen temporal wrapper
cb0 prediction head
causal depth decoder
32-codebook loss
Mimi decode validation path
```

Frame embedding:

```text
frame_embedding =
  E0(cb0) + E1(cb1) + ... + E31(cb31)
```

## Data Requirements

Primary target data:

```text
LibriTTS-R or equivalent TTS-quality speech corpus
24 kHz audio preferred
sentence-level text
speaker_id available
Mimi codes extracted with 32 codebooks
```

Expected row shape:

```text
id: string
text: string
speaker_id: int
codes: int[32, n_frames]
n_frames: int
k_codebooks: 32
```

The current 8-codebook pre-tokenized dataset is useful for smoke tests only. The serious Project 1 target needs 32-codebook codes.

## Data Pipeline Tasks

Required:

```text
1. Confirm Mimi can encode and decode 32 codebooks locally.
2. Build an extraction script for source audio -> codes[32, n_frames].
3. Save the dataset in Hugging Face datasets or JSONL/Arrow format.
4. Add validation that decoded extracted codes reconstruct intelligible audio.
5. Track source audio path, text, speaker_id, duration, n_frames, and codebook count.
```

Extraction validation:

```text
decode original Mimi codes
measure duration consistency
listen to random samples
reject failed/empty/too-long examples
```

## Training Objective

Training input:

```text
text tokens
previous audio frame codes
```

Training targets:

```text
cb0 for next frame from temporal backbone
cb1-cb31 for the same frame from depth decoder
```

Loss:

```text
total_loss =
  cb0_loss * cb0_weight
  + mean(cb1_to_cb31_losses)
```

Starting weights:

```text
cb0_weight = 5.0
cb1-cb31 = 1.0 each
```

Use teacher forcing for both temporal and depth paths during training.

## Training Stages

### Stage 1A: Smoke Run

Purpose:

```text
prove forward pass, loss, generation, save/load, and Mimi decode
```

Suggested size:

```text
1K-5K samples
500-2K steps
short max frames
```

Exit criteria:

```text
loss decreases
generated codebooks do not collapse
full generated codes decode without runtime errors
model save/load works
```

### Stage 1B: Small TTS Run

Purpose:

```text
first intelligible speech
```

Suggested size:

```text
train_clean_100 scale or equivalent
20K-100K steps depending hardware
```

Exit criteria:

```text
decoded speech has recognizable words
duration roughly matches text length
no severe token collapse
manual samples are stable across prompts
```

### Stage 1C: Larger TTS Run

Purpose:

```text
usable base TTS checkpoint for downstream projects
```

Suggested data:

```text
LibriTTS-R train_clean_100 + train_clean_360
optional train_other_500 after quality checks
```

Exit criteria:

```text
speech is consistently intelligible
speaker/timbre is stable enough for downstream reuse
streamed frame generation works
checkpoint can be resumed and evaluated
```

## Streaming Inference

Offline generation can decode the full code sequence at the end. Streaming generation should emit frames incrementally:

```text
1. Encode text prompt.
2. Generate next frame cb0 with cached Qwen state.
3. Generate cb1-cb31 with depth decoder.
4. Append frame to Mimi decode buffer.
5. Decode/play audio chunks as soon as enough frames are available.
6. Continue until stop condition.
```

Target behavior:

```text
first audio under 300 ms eventually
stable frame rate near 12.5 fps
no full-sequence recomputation during generation
```

## Validation

Automated checks:

```text
loss by codebook
cb0 accuracy/perplexity
unique-token collapse report by codebook
duration prediction drift
save/load roundtrip
Mimi decode roundtrip
```

Manual checks:

```text
listen to fixed prompt set
compare short, medium, and long sentences
test punctuation and numbers
test multiple speakers if speaker conditioning is added
```

Collapse warning signs:

```text
same token repeated across many frames
same full frame repeated repeatedly
very low unique token count per codebook
decoded output is only static/noise after real training
```

## Deliverables

Required outputs:

```text
32-codebook Mimi extraction script
Mini-Miso TTS model code
training notebook or script
save/load script
generation script
Mimi decode test
small trained checkpoint
sample WAV outputs
training report with loss and sample notes
```

Checkpoint name:

```text
qwen-mini-miso-tts-32cb
```

## Main Risks

```text
32-codebook extraction is expensive.
Qwen3 0.6B may be too small for high-quality prosody.
Depth decoder may need more capacity than expected.
Mimi decode API details may differ between package versions.
Generated codes can look non-collapsed but still decode poorly.
```

## Success Definition

Project 1 is successful when the model can take text and generate decoded audio that is recognizably speech, with stable duration and no severe collapse, using the Qwen temporal backbone plus 32-codebook depth decoder.
