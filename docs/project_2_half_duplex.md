# Project 2: Half-Duplex Speech Dialogue

## Goal

Turn the Project 1 streaming TTS model into a clean turn-taking speech dialogue model.

The model should:

```text
listen to the user
wait through the user turn
generate a spoken assistant answer
stay silent while the user is speaking
```

This project is half-duplex. The assistant does not need to speak while listening yet.

Non-goals:

```text
no backchannels during user speech
no overlap handling
no interruption behavior
no simultaneous listen/speak
```

Those are Project 3.

## Starting Checkpoint

Project 2 starts from:

```text
qwen-mini-miso-tts-32cb
```

This checkpoint should already know:

```text
text -> Mimi speech codes
32-codebook depth decoding
streaming frame generation
Mimi waveform decode
```

Project 2 adds dialogue timing and response conditioning.

## Architecture

Use the same Mini-Miso audio generator:

```text
Qwen temporal backbone
+ 32-codebook depth decoder
+ Mimi decoder
```

Add a half-duplex dialogue frame representation:

```text
time frame t:
  user stream:      user cb0-cb31 or silence
  assistant stream: assistant cb0-cb31 or silence
  optional text:    transcript / response text / control tokens
```

The temporal backbone sees previous user and assistant frames and predicts assistant output.

Conceptual runtime:

```text
user speaks
  ->
Mimi encoder produces user codes
  ->
Qwen tracks the user turn
  ->
assistant remains silent
  ->
user stops
  ->
Qwen starts assistant response
  ->
depth decoder generates assistant speech codes
```

## Data Requirements

Primary data:

```text
clean two-speaker conversations
no intentional overlap
aligned or alignable turn audio
speaker roles S0 and S1
text transcripts available
```

Project data can come from:

```text
ZipVoice-Dialog generated scripts
TTS-rendered two-speaker audio
real dialogue corpora if available
```

Expected processed row:

```text
conversation_id: string
turns: list
  speaker: S0 or S1
  text: string
  audio_path: string
  mimi_codes: int[32, n_frames]
  start_frame: int
  end_frame: int
```

For assistant training, usually map:

```text
S0 = user
S1 = assistant
```

## Frame Format

At every 80 ms Mimi frame:

```text
user_stream:
  user cb0-cb31 if S0 is speaking
  silence token otherwise

assistant_stream:
  assistant cb0-cb31 if S1 is speaking
  silence token otherwise
```

Example:

```text
time 0 ms:
  user:      speech
  assistant: silence

time 1600 ms:
  user:      silence
  assistant: speech

time 3200 ms:
  user:      speech
  assistant: silence
```

## Training Objective

Main targets:

```text
assistant speech/silence decision
assistant cb0 temporal prediction
assistant cb1-cb31 depth prediction
optional assistant transcript/text prediction
```

Loss:

```text
total_loss =
  assistant_speak_loss * high_weight
  + assistant_cb0_loss * cb0_weight
  + assistant_depth_loss
  + optional_text_loss
```

Starting weights:

```text
assistant_speak_loss: high
cb0_loss: 5x
cb1-cb31: 1x
optional text loss: 0.5x
```

The silence/speak decision matters because a speech model that always talks is unusable.

## Training Stages

### Stage 2A: Synthetic Clean Turns

Purpose:

```text
teach clean user-turn -> assistant-turn behavior
```

Data:

```text
generated ZipVoice-Dialog scripts
TTS-rendered user and assistant turns
fixed gap between turns
no overlap
```

Exit criteria:

```text
assistant stays silent during user audio
assistant starts after user ends
assistant produces response-like speech
no severe speaker leakage
```

### Stage 2B: Varied Turn Timing

Purpose:

```text
make turn-taking robust
```

Add:

```text
variable gaps
short acknowledgements as full turns
longer user turns
different speaker voices
background silence/noise
```

Exit criteria:

```text
reasonable start latency after user turn
low false-start rate during user speech
stable response duration
```

### Stage 2C: Streaming Half-Duplex Loop

Purpose:

```text
prove runtime path works with microphone-like chunks
```

Runtime:

```text
stream user audio into Mimi encoder
append user frames to context
predict assistant silence until user ends
generate assistant frames after endpoint
play decoded assistant audio
```

Exit criteria:

```text
usable end-to-end spoken exchange
assistant does not talk over user in normal cases
latency measured and logged
```

## Endpointing

Half-duplex can use either external endpointing or learned endpointing.

Recommended first version:

```text
external VAD/endpointer controls when assistant may start
model learns speech/silence but runtime is guarded
```

Later version:

```text
model predicts speak/silence directly
external VAD only acts as safety guard
```

## Validation

Automated metrics:

```text
assistant silence accuracy during user speech
assistant start latency after user end
false interruption rate
assistant codebook loss by codebook
decoded response duration
turn completion rate
```

Manual checks:

```text
short Q&A conversations
multi-turn follow-up conversations
long user question
ambiguous pause handling
speaker identity consistency
```

Failure cases:

```text
assistant starts too early
assistant waits forever
assistant talks through user turn
assistant produces speech-like noise instead of response
assistant repeats previous answer
```

## Deliverables

Required outputs:

```text
half-duplex dataset builder
two-stream frame formatter
half-duplex training script
streaming half-duplex inference demo
latency logging
evaluation prompt set
sample conversation WAVs
training report
```

Checkpoint name:

```text
qwen-mini-miso-half-duplex
```

## Main Risks

```text
Synthetic dialogue may teach unnatural timing.
Bad endpointing can hide model timing problems.
Assistant may learn to over-speak if silence examples are weak.
Dialogue response quality may be limited by Qwen3 0.6B.
Audio quality can regress from Project 1 if training is too aggressive.
```

## Success Definition

Project 2 is successful when the model can run a clean spoken turn-taking conversation: user speaks, assistant waits, assistant answers in speech, and the model remains stable over several turns.
