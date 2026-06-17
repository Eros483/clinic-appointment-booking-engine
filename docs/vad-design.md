# Voice Activity Detection — Silero VAD via `silero-vad` Package

**Date:** 2026-06-17  
**Status:** Implemented  
**Layer:** 2 (see `docs/design.md` §4)

---

## Overview

Voice Activity Detection (VAD) determines when a caller is speaking and when they have stopped. This drives the core turn-taking in the voice agent:

- Detecting when the caller starts speaking → begin buffering audio
- Detecting end-of-utterance → trigger the downstream pipeline (language ID → STT → LLM → TTS)
- Detecting speech during TTS playback → **barge-in** (interrupt the bot response)

We use [Silero VAD](https://github.com/snakers4/silero-vad) via its official PyPI package [`silero-vad`](https://pypi.org/project/silero-vad/). The package bundles the pre-trained model (both JIT and ONNX variants) and handles ONNX session lifecycle, GRU hidden state, and context window internally.

---

## Why `silero-vad` Package Instead of Raw ONNX?

| Approach | Pros | Cons |
|---|---|---|
| **`silero-vad` package** (chosen) | Zero model download code, ONNX session managed internally, GRU state + context window handled, `reset_states()` built-in, auto-updates with `pip` | Additional dependency |
| Raw ONNX (downloaded from GitHub) | No dependency on silero-vad | Manual download, manual state management, 64-sample context window must be handled, more boilerplate |

The package wraps the same ONNX model we would download manually — there is no trade-off in latency or accuracy. It eliminates ~100 lines of boilerplate.

---

## Architecture

### Files

| File | Role |
|---|---|
| `backend/core/vad.py` | `VADProcessor` class + module-level model loader |
| `backend/tests/test_vad.py` | 12 unit tests with mocked ONNX session |

### Module-Level Model Loader

```python
_model = None

def _load_model():
    from silero_vad import load_silero_vad
    _model = load_silero_vad(onnx=True)  # OnnxWrapper instance
```

- Lazy-loaded on first `process()` call (cold start)
- `load_silero_vad(onnx=True)` returns an `OnnxWrapper` that manages the ONNX session, GRU state, and 64-sample context window
- Falls back gracefully if `silero-vad` is not installed (returns `0.0` for all speech probabilities)

### VADProcessor Class

One instance per Twilio call. Maintains:

- **`state`** — `"idle"` | `"speaking"`
- **`silence_counter`** — consecutive frames with `speech_prob <= 0.5`
- **`pre_roll`** — circular buffer of the last ~300ms of audio (19 frames)
- **`utterance`** — accumulated frames for the current utterance

---

## Decision Logic

### State Machine

```
                  speech_prob > 0.5 + TTS active
                  ┌─────────────────────────────┐
                  │                             │
                  ▼                             │
┌────────┐  speech_prob > 0.5  ┌──────────┐    │
│  IDLE  │────────────────────►│ SPEAKING │────┘
│        │◄────────────────────│          │
└────────┘  silence_counter     └──────────┘
             >= 44 frames
```

### Per-Frame Events

| Condition | Event | Action |
|---|---|---|
| `prob > 0.5`, state is `idle` | `VAD_START` | Prepend pre-roll buffer, start accumulating frames |
| `prob > 0.5`, state is `speaking` | `None` | Append frame to utterance buffer |
| `prob <= 0.5`, state is `speaking`, silence < 44 frames | `None` | Append frame, increment counter |
| `prob <= 0.5`, state is `speaking`, silence >= 44 frames | `VAD_END` | Reset state, return accumulated utterance |
| `prob > 0.5`, TTS is active | `BARGE_IN` | Reset model + utterance buffer, prepend pre-roll |

### Constants

| Constant | Value | Rationale |
|---|---|---|
| `SAMPLE_RATE` | 16000 Hz | Twilio audio is 8kHz μ-law, ups sampled to 16kHz pre-VAD |
| `FRAME_SIZE` | 512 samples | 32ms per frame at 16kHz — Silero VAD standard |
| `SPEECH_PROB_THRESHOLD` | 0.5 | Silero default; speech probability above this = speech |
| `SILENCE_THRESHOLD_FRAMES` | 44 | ~704ms of consecutive silence → end of utterance |
| `PRE_ROLL_FRAMES` | 19 | ~304ms circular buffer before speech onset |

---

## Pre-Roll Buffer

The pre-roll buffer is a circular list that always holds the last `PRE_ROLL_FRAMES` (19) frames of audio:

```python
def _update_pre_roll(self, frame):
    self.pre_roll.append(frame)
    if len(self.pre_roll) > PRE_ROLL_FRAMES:
        self.pre_roll.pop(0)
```

When `VAD_START` fires, the pre-roll contents are prepended to the utterance buffer:

```python
def _on_speech_start(self, frame):
    self.utterance = list(self.pre_roll) + [frame]
    return VAD_START
```

This captures the first ~300ms of speech that occurred before the VAD probability crossed the threshold — crucial for language ID and STT which need the full utterance onset.

---

## Barge-In

When the caller interrupts the bot during TTS playback:

1. `_model.reset_states()` — clears Silero's GRU hidden state (fresh context)
2. Utterance buffer is flushed and replaced with `pre_roll + [current_frame]`
3. Returns `BARGE_IN` event

The caller (state machine or WebSocket handler) should:
- Kill the TTS audio queue
- Send a Twilio `<Stop>` command
- Begin collecting the new utterance

---

## Testing Strategy

12 unit tests in `backend/tests/test_vad.py`, all using `unittest.mock.patch` to replace the module-level `_model` with a `MagicMock` returning controlled speech probabilities.

| # | Test | Coverage |
|---|---|---|
| 1 | `test_vad_start_on_speech_after_silence` | `VAD_START` on first speech frame after idle |
| 2 | `test_vad_end_after_silence_threshold` | `VAD_END` after 44 consecutive silence frames |
| 3 | `test_barge_in_during_tts` | `BARGE_IN` when `is_tts_active=True` |
| 4 | `test_pre_roll_buffer_captured_on_vad_start` | Pre-roll prepended on VAD_START |
| 5 | `test_utterance_accumulates_during_speech` | Multiple speech frames appended |
| 6 | `test_get_utterance_audio_returns_concatenated` | Correct shape and dtype |
| 7 | `test_get_utterance_audio_empty_when_no_speech` | Empty array before any speech |
| 8 | `test_reset_clears_all_state` | All state + `_model.reset_states()` called |
| 9 | `test_short_burst_does_not_emit_early_end` | Single silence frame is not VAD_END |
| 10 | `test_full_idle_speaking_idle_cycle` | Complete round-trip |
| 11 | `test_raises_on_wrong_frame_size` | ValueError on non-512 frames |
| 12 | `test_barge_in_flushes_previous_utterance` | `_model.reset_states()` called on barge-in |

---

## Integration Points

### Upstream: Audio Processing

Twilio delivers 8kHz μ-law frames. These are decoded and resampled to 16kHz PCM in `backend/core/audio.py` before being passed to `VADProcessor.process()`.

### Downstream: Pipeline Trigger

On `VAD_END`:
1. Call `get_utterance_audio()` to get the accumulated audio
2. Pass to language ID (`backend/core/language_id.py`)
3. Pass to STT (`backend/core/stt.py`)
4. Pass transcript to LLM orchestration

On `BARGE_IN`:
1. Kill TTS playback
2. Flush any pending pipeline
3. Begin collecting new utterance

### State

`VADProcessor` is independent per Twilio call. Create a new instance in the call's session initialization.

---

## Dependencies

| Package | Version | Notes |
|---|---|---|
| `silero-vad` | ≥ 6.2.1 | Bundles model, provides `load_silero_vad()` + `OnnxWrapper` |
| `onnxruntime` | ≥ 1.19.0 | Required by `silero-vad` for ONNX inference |
| `torch` | ≥ 2.4.0 | Required by `silero-vad` (used for tensor ops, not CUDA) |
| `numpy` | ≥ 1.26.0 | Frame and buffer handling |

---

## Usage Example

```python
from backend.core.vad import VADProcessor, VAD_START, VAD_END, BARGE_IN

vad = VADProcessor()

# Simulate streaming: silence then speech then silence
for frame in audio_stream:
    event = vad.process(frame, is_tts_active=tts_playing)

    if event == VAD_START:
        print("Caller started speaking")

    elif event == VAD_END:
        audio = vad.get_utterance_audio()
        print(f"Caller finished: {len(audio)} samples")
        vad.reset()  # ready for next utterance

    elif event == BARGE_IN:
        print("Caller interrupted!")
        kill_tts()
```

---

## Tunables

| Parameter | Location | Effect |
|---|---|---|
| `SPEECH_PROB_THRESHOLD` (0.5) | `vad.py` | Lower → more sensitive to noise; higher → may miss soft speech |
| `SILENCE_THRESHOLD_FRAMES` (44) | `vad.py` | Lower → faster turn-taking, risk of mid-utterance cutoff |
| `PRE_ROLL_FRAMES` (19) | `vad.py` | Larger → more context before VAD_START, more latency |
