# Project 3: Full-Duplex Speech Dialogue

## Goal

Extend the half-duplex speech dialogue model into a full-duplex model that can listen and speak at the same time.

The model should handle:

```text
backchannels while the user continues speaking
natural short overlaps
assistant stop/yield when user interrupts
user interruptions during assistant speech
simultaneous starts
continued listening while assistant audio is being generated
```

This is the final realtime conversation target.

## Starting Checkpoint

Project 3 starts from:

```text
qwen-mini-miso-half-duplex
```

That checkpoint should already know:

```text
speech token generation
32-codebook Mimi depth decoding
clean turn-taking
user/assistant stream formatting
streaming inference
```

Project 3 adds overlap and simultaneous listen/speak behavior.

## Architecture

Use the same Mini-Miso model family:

```text
Qwen temporal backbone
+ 32-codebook depth decoder
+ frozen Mimi decoder
```

Full-duplex runtime uses two audio streams at every frame:

```text
time frame t:
  user_stream:      user cb0-cb31 or silence
  assistant_stream: assistant cb0-cb31 or silence
```

The key difference from half-duplex:

```text
user_stream and assistant_stream can both contain speech in the same frame
```

Example:

```text
time 2400 ms:
  user:      speaking
  assistant: backchannel

time 2480 ms:
  user:      speaking
  assistant: still backchanneling

time 2560 ms:
  user:      speaking
  assistant: silence
```

## Runtime Loop

Full-duplex inference:

```text
1. Capture microphone PCM continuously.
2. Encode user audio into Mimi frames.
3. Feed user frames into the model even while assistant is speaking.
4. Predict assistant speak/silence/yield decisions every frame.
5. Generate assistant cb0-cb31 when speaking.
6. Decode assistant frames to audio.
7. Play assistant audio.
8. Feed previous assistant frames back into model context.
9. Continue without waiting for hard user-turn boundaries.
```

Unlike half-duplex, there is no strict "wait until user stops" rule.

## Data Requirements

Primary data:

```text
overlapped two-speaker dialogue
backchannel examples
interruptions
stop/yield examples
simultaneous starts
clean non-overlap examples retained for stability
```

Processed row:

```text
conversation_id: string
timeline_frames: list
  user_codes: int[32] or silence
  assistant_codes: int[32] or silence
  user_speaking: bool
  assistant_speaking: bool
  event_label: optional
```

Useful labels:

```text
normal_turn
backchannel
interruption
yield
continue_speaking
simultaneous_start
silence
```

Labels are optional for audio generation but useful for weighted losses and debugging.

## Overlap Types

Backchannels:

```text
user continues speaking
assistant briefly says "yeah", "right", "mm-hm"
assistant returns to silence
```

Interruptions:

```text
assistant starts before user finishes
or user interrupts assistant
```

Yield behavior:

```text
assistant is speaking
user starts speaking
assistant stops or shortens response
```

Simultaneous starts:

```text
both streams begin almost together
assistant should usually yield unless it has strong reason to continue
```

## Training Objective

Main targets:

```text
assistant speak/silence/backchannel/yield decision
assistant cb0 temporal prediction
assistant cb1-cb31 depth prediction
assistant stop behavior during user interruption
```

Loss:

```text
total_loss =
  assistant_event_loss
  + assistant_cb0_loss * cb0_weight
  + assistant_depth_loss
  + stop_yield_loss * high_weight
  + optional text/control loss
```

Recommended weighting:

```text
cb0_loss: 5x
cb1-cb31: 1x
stop/yield examples: upweight
backchannel examples: upweight moderately
false-overlap penalty: high
```

## Training Stages

### Stage 3A: Controlled Backchannels

Purpose:

```text
teach short assistant speech while user continues
```

Data:

```text
mostly normal user speech
short assistant backchannels inserted at plausible points
```

Exit criteria:

```text
assistant can produce short backchannels
assistant returns to silence
assistant does not take over the floor
```

### Stage 3B: Interrupt And Yield

Purpose:

```text
teach the model to stop or yield when user cuts in
```

Data:

```text
assistant speaking
user interruption starts
assistant stops within a short window
```

Exit criteria:

```text
low failure-to-stop rate
assistant does not continue long monologues through user speech
audio remains stable during stop
```

### Stage 3C: Natural Overlap Mix

Purpose:

```text
combine clean turns, backchannels, interruptions, and silence
```

Data mix:

```text
clean half-duplex examples retained
controlled backchannels
interrupt/yield examples
real overlap examples if available
noise/echo augmented examples
```

Exit criteria:

```text
stable multi-turn full-duplex demo
reasonable overlap behavior
no constant talking
no constant silence
no collapse during simultaneous speech
```

## Echo And Noise Augmentation

Full-duplex microphones hear the assistant speaker through the room. Training must include this.

Augmentations:

```text
assistant-output echo in user channel
room reverb
background noise
microphone gain variation
small timing jitter
speaker volume differences
```

Without echo augmentation, a model may break when deployed with real speakers and microphones.

## Validation

Automated metrics:

```text
backchannel timing rate
false interruption rate
failure-to-stop rate
assistant speech ratio
silence ratio
overlap duration distribution
decoded audio quality during overlap
latency from user interruption to assistant stop
```

Manual tests:

```text
user speaks long explanation
assistant gives short backchannel
user interrupts assistant
assistant yields
user pauses briefly
assistant does not overreact
noisy room test
speaker echo test
```

Failure cases:

```text
assistant constantly backchannels
assistant never backchannels
assistant refuses to stop
assistant stops too easily
assistant speech degrades during overlap
model confuses user and assistant streams
```

## Streaming Requirements

Full-duplex must be evaluated in a real streaming loop:

```text
microphone input
Mimi encoder
frame scheduler
Qwen temporal cache
depth decoder
Mimi decoder
speaker output
assistant audio feedback into context
```

Target latency:

```text
frame size: 80 ms
first assistant audio: 180-320 ms target
stop/yield reaction: under 300 ms target
```

## Deliverables

Required outputs:

```text
full-duplex frame dataset builder
overlap/backchannel data generator
echo/noise augmentation pipeline
full-duplex training script
streaming full-duplex runtime demo
latency and interruption metrics
sample interactive recordings
evaluation report
```

Checkpoint name:

```text
qwen-mini-miso-full-duplex
```

## Main Risks

```text
Good full-duplex data is hard to obtain.
Synthetic overlap may teach unnatural behavior.
Assistant can become too eager to speak.
Echo leakage can destabilize the model.
Stop/yield behavior may damage normal answer completion.
Qwen3 0.6B may not have enough capacity for robust realtime dialogue.
```

## Success Definition

Project 3 is successful when the model can run a live spoken conversation where it listens continuously, speaks when appropriate, gives short backchannels, and yields when the user interrupts, without collapsing into constant speech or silence.
