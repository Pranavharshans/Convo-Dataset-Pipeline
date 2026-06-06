# Real-Time Duplex Training Plan

This document describes the planned training path for a real-time full-duplex speech model using:

```text
Qwen 0.6B temporal backbone
Mimi streaming audio codec
32-codebook Mini-Miso depth decoder target
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

## Project Specs

The roadmap is split into three standalone project specs:

```text
Project 1: Streaming Mini-Miso TTS
  docs/project_1_streaming_tts.md

Project 2: Half-Duplex Speech Dialogue
  docs/project_2_half_duplex.md

Project 3: Full-Duplex Speech Dialogue
  docs/project_3_full_duplex.md
```

## Architecture Reference: MisoTTS

MisoTTS is a useful validated reference for this plan because it uses the same basic direction:

```text
large temporal backbone over audio/text time
+ small depth decoder over Mimi codebooks
+ Mimi decoder to waveform
```

The public MisoTTS 8B release uses:

```text
7.7B temporal backbone
300M audio depth decoder
Mimi codec
32 codebooks
half-duplex turn generation
```

This does not mean we should copy the 8B model size. It means the temporal/depth split is the right architecture to copy.

For this project, the practical version is:

```text
Qwen3 0.6B or 1.7B temporal backbone
small 50M-150M depth decoder
32 Mimi codebooks for the main TTS target
8 Mimi codebooks only for cheap smoke tests and existing pre-tokenized data
```

The current Unsloth notebook with 8 parallel codebook heads is a smoke test only. It is useful for proving the data path and loss are wired correctly, but it should not be treated as the final realtime architecture.

Updated project target:

```text
1. Mini-Miso TTS:
   Qwen3 0.6B + depth decoder + 32 Mimi codebooks
   no dialogue, no duplex

2. Mini-Miso half-duplex dialogue:
   same model family
   clean turn-based speaker response

3. Mini-Miso full-duplex:
   two-stream user/assistant frame model
   overlap, backchannels, interruption, stop/yield
```

This ladder is easier to debug than jumping directly to full duplex. Each stage isolates one hard problem.

## Core Architecture

Do not use flattened codebook tokens as the final real-time architecture.

Flattened 8-codebook audio looks like this:

```text
frame 0: cb0 cb1 cb2 cb3 cb4 cb5 cb6 cb7
frame 1: cb0 cb1 cb2 cb3 cb4 cb5 cb6 cb7
```

Mimi runs at 12.5 frames/sec. Flattening 8 codebooks gives:

```text
12.5 frames/sec * 8 codebooks = 100 autoregressive tokens/sec
```

Flattening 32 codebooks would be worse:

```text
12.5 frames/sec * 32 codebooks = 400 autoregressive tokens/sec
```

That is expensive and increases latency.

The real-time architecture should follow the Moshi/Miso-style split:

```text
Qwen temporal backbone:
  runs once per 80 ms Mimi frame
  tracks language, dialogue context, timing, and turn behavior

Small depth transformer:
  predicts the within-frame codebook stack
  conditions cb1 on cb0
  conditions cb2 on cb0-cb1
  ...
  conditions cb31 on cb0-cb30 for the 32-codebook model
```

This keeps the large model near:

```text
12.5 big-model steps/sec
```

instead of:

```text
100 big-model steps/sec
```

Recommended frame embedding:

```text
frame_embedding =
  E0(cb0) + E1(cb1) + E2(cb2) + ... + E31(cb31)
```

For text-only positions, audio codebooks are masked. For audio positions, text tokens can be masked or represented through a separate text stream. This keeps one temporal position per Mimi frame instead of one temporal position per codebook.

Recommended prediction path:

```text
temporal backbone hidden[t]
  -> cb0 logits for frame t
  -> depth decoder input: hidden[t] + cb0
  -> cb1 logits
  -> depth decoder input: hidden[t] + cb0 + cb1
  -> cb2 logits
  ...
  -> cb31 logits
```

During training, use teacher forcing for the depth decoder. During inference, generate codebooks autoregressively inside the frame.

## Mimi Codebooks

Mimi can be used at different codebook depths. The existing pre-tokenized dataset gives 8 codebooks per frame:

```text
codes: [8, n_frames]
```

The Mini-Miso TTS target should use 32 codebooks if we can get or extract compatible codes:

```text
codes: [32, n_frames]
```

Useful mental model:

```text
cb0: strongest speech-content / semantic signal
cb1-cb31: acoustic detail, timbre, quality, reconstruction detail
```

For the final TTS model, use 32 codebooks. A one-codebook model may learn rough speech content, and an 8-codebook model may be useful for early experiments, but 32 codebooks gives a better quality ceiling for natural speech.

MisoTTS uses 32 Mimi codebooks, but the current LibriTTS-R Mimi dataset has 8:

```text
codes: [8, n_frames]
k_codebooks: 8
```

For the updated target, do not treat the 8-codebook dataset as the final data source. Use it to validate code, but plan to either re-extract 32-codebook Mimi codes or find a compatible pre-tokenized 32-codebook dataset before serious TTS quality training.

For training loss, weight cb0 more strongly:

```text
cb0 loss:    5-10x
cb1-cb31:    1x
text loss:   0.5-1x when text supervision is present
silence/speak loss: high in dialogue stages
```

The exact weights should be tuned by validation decode quality and turn behavior.

## Stage 1: Mini-Miso TTS Pretraining

Goal:

```text
Teach Qwen + depth decoder to generate clean Mimi speech tokens from text.
```

Primary dataset target:

```text
LibriTTS-R or equivalent TTS-quality speech dataset
Mimi codes extracted at 32 codebooks
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
32 codebooks available after extraction
```

If only 8-codebook precomputed codes are available, use them for smoke tests and loader validation. For the actual Mini-Miso TTS checkpoint, extract or acquire 32-codebook Mimi codes.

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

Stage 1 should train the real Mini-Miso temporal/depth architecture, not the old parallel-head architecture:

```text
Input:
  transcript text
  previous Mimi frames

Temporal target:
  next-frame cb0

Depth targets:
  cb1-cb31 for the same frame
```

Parallel heads are acceptable only for a quick smoke test:

```text
Qwen hidden -> 8 independent heads
```

The 8-codebook smoke-test version uses:

```text
Qwen hidden -> cb0 head -> depth decoder -> cb1-cb7
```

For the 32-codebook Mini-Miso target, this becomes:

```text
Qwen hidden -> cb0 head -> depth decoder -> cb1-cb31
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
qwen-mini-miso-tts-32cb
```

This checkpoint should generate intelligible single-speaker TTS. It should not yet be expected to handle dialogue timing.

## Stage 2: Half-Duplex Dialogue Training

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
  user cb0-cb31 if user is speaking
  silence token if user is not speaking

assistant stream:
  assistant cb0-cb31 if assistant is speaking
  silence token if assistant is not speaking

optional text stream:
  assistant inner-monologue or transcript tokens
```

Example:

```text
time 0 ms:
  user:      speech cb0-cb31
  assistant: silence

time 1600 ms:
  user:      silence
  assistant: speech cb0-cb31

time 3200 ms:
  user:      speech cb0-cb31
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
qwen-mini-miso-half-duplex
```

This checkpoint should behave like a stable turn-taking speech model. It should listen, wait, then answer. It does not need simultaneous listen/speak behavior yet.

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
  user:      speaking cb0-cb31
  assistant: backchannel cb0-cb31

time 2480 ms:
  user:      speaking cb0-cb31
  assistant: still speaking cb0-cb31

time 2560 ms:
  user:      speaking cb0-cb31
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
qwen-mini-miso-full-duplex
```

This is the first checkpoint that should be tested in a streaming microphone/speaker loop.

## Inference Loop

Runtime path:

```text
1. Capture microphone PCM.
2. Stream into Mimi encoder.
3. Append user Mimi tokens to the user stream.
4. Run Qwen temporal step every Mimi frame.
5. Predict assistant cb0-cb31 through the depth decoder.
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
1. Keep the current 8-codebook notebook only as a smoke test.
2. Build or acquire 32-codebook Mimi token extraction for LibriTTS-R or equivalent.
3. Build the Mini-Miso model: Qwen temporal backbone + 32-codebook depth decoder.
4. Train Stage 1 TTS until decoded samples are intelligible and stable.
5. Add cached frame-by-frame generation for low-latency TTS.
6. Build frame-stream representation for user/assistant/silence.
7. Train Stage 2 half-duplex dialogue until clean turn-taking is stable.
8. Train Stage 3 full-duplex overlap behavior.
9. Build streaming microphone/speaker inference loop.
10. Measure actual end-to-end latency and failure modes.
```

Avoid optimizing the full-duplex runtime before Stage 1 decoded audio works. If Stage 1 cannot produce intelligible speech tokens, later dialogue stages will not fix it.

## Checkpoint Ladder

```text
qwen-base-0.6b
  -> qwen-mini-miso-tts-32cb
  -> qwen-mini-miso-half-duplex
  -> qwen-mini-miso-full-duplex
  -> qwen-mini-miso-instruction-duplex
```

Each checkpoint should be saved separately. This makes it possible to roll back if a later stage damages audio quality or language behavior.

## Main Risks

```text
Qwen 0.6B may be too small for high-quality long dialogue.
32-codebook extraction is more expensive than using the current 8-codebook dataset.
Flattened-codebook prototypes may look okay offline but fail latency targets.
Bad silence modeling can make the assistant talk too much.
Weak cb0 prediction can destroy intelligibility.
Noisy overlap data can teach bad interruption behavior.
Lack of echo augmentation can break real microphone deployment.
```

The fastest path is still:

```text
Mimi + Qwen temporal model + small 32-codebook depth decoder
```

Do not switch codecs unless Mimi becomes a hard blocker.
