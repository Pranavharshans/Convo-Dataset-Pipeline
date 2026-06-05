# Real-Time Duplex Training Plan

This document describes the planned training path for a real-time full-duplex speech model using:

```text
Qwen 0.6B temporal backbone
Mimi streaming audio codec
8-codebook depth predictor
ZipVoice-Dialog generated conversations
Stage 4 overlap/backchannel data
```

The goal is not a turn-based TTS chatbot. The goal is a model that can continuously listen and speak with low latency:

```text
microphone PCM
-> Mimi encoder
-> user audio-token stream
-> Qwen temporal model
-> assistant Mimi codebooks
-> Mimi decoder
-> speaker audio
```

No external ASR is required at inference. Transcripts are used during training because they help the model connect speech tokens to language.

## Core Architecture

Do not use flattened 8-codebook tokens as the final real-time architecture.

Flattened audio looks like this:

```text
frame 0: cb0 cb1 cb2 cb3 cb4 cb5 cb6 cb7
frame 1: cb0 cb1 cb2 cb3 cb4 cb5 cb6 cb7
```

Mimi runs at 12.5 frames/sec. Flattening 8 codebooks gives:

```text
12.5 frames/sec * 8 codebooks = 100 autoregressive tokens/sec
```

That is expensive and increases latency.

The real-time architecture should follow the Moshi-style split:

```text
Qwen temporal backbone:
  runs once per 80 ms Mimi frame
  tracks language, dialogue context, timing, and turn behavior

Small depth transformer or codebook predictor:
  predicts cb0-cb7 for the current assistant frame
```

This keeps the large model near:

```text
12.5 big-model steps/sec
```

instead of:

```text
100 big-model steps/sec
```

## Mimi Codebooks

Mimi gives 8 codebooks per frame:

```text
codes: [8, n_frames]
```

Useful mental model:

```text
cb0: strongest speech-content / semantic signal
cb1-cb7: acoustic detail, timbre, quality, reconstruction detail
```

For the final model, use all 8 codebooks. A one-codebook model may learn rough speech content, but it will not be enough for clean natural audio reconstruction.

For training loss, weight cb0 more strongly:

```text
cb0 loss:    5-10x
cb1-cb7:     1x
text loss:   0.5-1x when text supervision is present
silence/speak loss: high in dialogue stages
```

The exact weights should be tuned by validation decode quality and turn behavior.

## Stage 1: Speech Audio Pretraining

Goal:

```text
Teach Qwen + depth predictor to understand and generate Mimi speech tokens.
```

Primary dataset:

```text
LibriTTS-R Mimi Codes
```

Expected row shape:

```text
text: "He hoped there would be stew for dinner."
codes: [8, n_frames]
speaker_id: int
n_frames: int
```

Why this dataset is useful:

```text
24 kHz audio
sentence-level segmentation
punctuation preserved
Mimi codes already extracted
all 8 codebooks available
```

No expensive re-tokenization is needed. Load the precomputed `codes` and train directly.

Training task:

```text
transcript/text conditioning -> Mimi audio codebooks
```

Conceptual sample:

```text
text stream:
  He hoped there would be stew for dinner.

audio frames:
  frame 0: cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7
  frame 1: cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7
  ...
```

What the model should learn:

```text
speech-token rhythm
text-to-speech alignment
basic pronunciation
Mimi codebook prediction
short-form speech generation
```

Recommended batching:

```text
audio/text batches: majority
text-only Qwen batches: 30-50% early if language forgetting appears
```

Validation checks:

```text
audio loss by codebook
cb0 accuracy/perplexity
decoded speech intelligibility
duration stability
language retention on text-only evals
```

Output checkpoint:

```text
qwen-mimi-audio-base
```

This checkpoint should know speech tokens, but it should not yet be expected to handle natural conversation timing.

## Stage 2: Clean Conversation Training

Goal:

```text
Teach turn-taking, two-speaker timing, and response generation.
```

Primary dataset:

```text
50K ZipVoice-Dialog conversations
about 450 hours total
no intentional overlap
```

Conversation split:

```text
short:    15K, 4-6 turns, quick exchanges
medium:   20K, 6-8 turns, natural flow
long:     10K, 8-10 turns, sustained topic
extended:  5K, 10-14 turns, story arcs / deeper explanations
```

Stage 2 should create aligned frame streams. At every 80 ms Mimi frame:

```text
user stream:
  user cb0-cb7 if user is speaking
  silence token if user is not speaking

assistant stream:
  assistant cb0-cb7 if assistant is speaking
  silence token if assistant is not speaking

optional text stream:
  assistant inner-monologue or transcript tokens
```

Example:

```text
time 0 ms:
  user:      speech cb0-cb7
  assistant: silence

time 1600 ms:
  user:      silence
  assistant: speech cb0-cb7

time 3200 ms:
  user:      speech cb0-cb7
  assistant: silence
```

What the model should learn:

```text
listen while silent
start after the other speaker finishes
avoid interrupting too early
preserve speaker identity
respond with coherent speech
emit silence when it should not speak
```

Training role setup:

```text
S0 = user
S1 = assistant
```

For broader pretraining, roles can be randomized. For assistant instruction tuning, lock the role mapping.

Recommended losses:

```text
assistant audio codebook loss: high
assistant silence/speak decision: high
assistant text/inner-monologue loss: useful
user audio reconstruction/prediction: optional or lower
```

Silence is important. The model must learn that not speaking is an active output decision, not missing data.

Validation checks:

```text
turn-taking latency
early interruption rate
silence prediction accuracy
assistant decoded audio quality
response coherence
speaker leakage between streams
```

Output checkpoint:

```text
qwen-mimi-clean-dialogue
```

This checkpoint should behave like a stable turn-taking speech model. It may still be too turn-based for full duplex.

## Stage 3: Overlap And Full-Duplex Training

Goal:

```text
Teach simultaneous listen/speak behavior.
```

Primary dataset:

```text
Stage 4 overlapped dialogue
50-200 hours
```

Overlap types:

```text
backchannels:
  "yeah", "right", "mm-hm" while the user continues speaking

interruptions:
  assistant or user cuts in before the other speaker fully ends

simultaneous starts:
  both streams begin at nearly the same time

self-correction / stop behavior:
  assistant stops when user cuts in
```

Frame representation:

```text
time 2400 ms:
  user:      speaking cb0-cb7
  assistant: backchannel cb0-cb7

time 2480 ms:
  user:      speaking cb0-cb7
  assistant: still speaking cb0-cb7

time 2560 ms:
  user:      speaking cb0-cb7
  assistant: silence
```

What the model should learn:

```text
listen while speaking
emit short backchannels without taking the floor
interrupt only when appropriate
stop or yield when the user starts speaking
continue through mild user noise
handle natural overlap windows
```

Recommended augmentations:

```text
room reverb
background noise
microphone variation
assistant-output echo in the user channel
small timing jitter
volume variation across speakers
```

Echo/noise augmentation matters because real full-duplex inference will hear microphone leakage and room sound.

Recommended losses:

```text
assistant audio codebook loss
assistant silence/speak decision loss
backchannel timing loss or weighted examples
stop/yield behavior examples
optional assistant text/inner-monologue loss
```

Validation checks:

```text
latency from user pause to assistant start
backchannel timing quality
false interruption rate
failure to stop when user interrupts
duplex stability during overlap
decoded speech quality during overlap
```

Output checkpoint:

```text
qwen-mimi-full-duplex
```

This is the first checkpoint that should be tested in a streaming microphone/speaker loop.

## Inference Loop

Runtime path:

```text
1. Capture microphone PCM.
2. Stream into Mimi encoder.
3. Append user Mimi tokens to the user stream.
4. Run Qwen temporal step every Mimi frame.
5. Predict assistant cb0-cb7 through the depth predictor.
6. Stream assistant codebooks into Mimi decoder.
7. Play decoded audio.
8. Feed previous assistant audio tokens back into context.
```

Target latency budget:

```text
Mimi frame:      80 ms
acoustic delay:  80-160 ms
model compute:   20-80 ms
target total:   ~180-320 ms
```

The model should be evaluated in streaming mode, not only by offline loss.

## Practical Development Order

Recommended implementation order:

```text
1. Build dataset loaders for precomputed LibriTTS-R Mimi codes.
2. Build frame-stream representation for user/assistant/silence.
3. Add Qwen 0.6B temporal backbone wrapper.
4. Add depth predictor for 8 Mimi codebooks.
5. Train Stage 1 until decoded samples are intelligible.
6. Train Stage 2 until clean turn-taking is stable.
7. Train Stage 3 until overlap behavior is usable.
8. Build streaming inference loop.
9. Measure actual end-to-end latency and failure modes.
```

Avoid optimizing the full-duplex runtime before Stage 1 decoded audio works. If Stage 1 cannot produce intelligible speech tokens, later dialogue stages will not fix it.

## Checkpoint Ladder

```text
qwen-base-0.6b
  -> qwen-mimi-audio-base
  -> qwen-mimi-clean-dialogue
  -> qwen-mimi-full-duplex
  -> qwen-mimi-instruction-duplex
```

Each checkpoint should be saved separately. This makes it possible to roll back if a later stage damages audio quality or language behavior.

## Main Risks

```text
Qwen 0.6B may be too small for high-quality long dialogue.
Flattened-codebook prototypes may look okay offline but fail latency targets.
Bad silence modeling can make the assistant talk too much.
Weak cb0 prediction can destroy intelligibility.
Noisy overlap data can teach bad interruption behavior.
Lack of echo augmentation can break real microphone deployment.
```

The fastest path is still:

```text
Mimi + Qwen 0.6B temporal model + small 8-codebook depth predictor
```

Do not switch codecs unless Mimi becomes a hard blocker.
