# Layer 3 — Language Identification — ✅ DONE

## Files created

### ✅ `backend/core/language_id.py`

Module-level model loader + two pure functions.

**`VOXLINGUA_TO_CODE` dict** — maps VoxLingua107 labels to 5 language codes:
- `Hindi` → `hi`, `English` → `en`, `Tamil` → `ta`, `Bengali` → `bn`, `Marathi` → `mr`

**_classifier** — module-level sentinel (`None`), lazy-loaded on first call via `EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa")`.

**`identify_language(audio: np.ndarray, sr: int = 16000) -> tuple[str, float]`**
1. Load `_classifier` if not loaded yet
2. Convert `audio` to `torch.tensor`, unsqueeze to shape `(1, T)`
3. Call `_classifier.classify_batch(signal)` → get label at `text_lab[0]`, confidence at `score[0].exp()`
4. Handle `"code: name"` format by splitting on `": "`
5. Map label via `VOXLINGUA_TO_CODE`, defaulting to `"hi"` on unrecognised labels
6. Return `(lang_code, confidence)`

**`update_active_language(prediction: str, confidence: float, word_count: int, current_language: str, is_first_utterance: bool) -> str`**
- If `is_first_utterance` and `confidence < 0.80` → return `"hi"`
- If `confidence >= 0.80` and `word_count > 5` → return `prediction` (switch)
- Otherwise → return `current_language` (keep)

---

### ✅ `backend/scripts/eval_lang_id.py`

Standalone script. Import paths: `speechbrain`, `torch`, `datasets`, `numpy`, `scipy`, `librosa`, `time`.

1. Load `speechbrain/lang-id-voxlingua107-ecapa` classifier
2. Sample 50 utterances × 5 languages (250 total) from `ai4bharat/IndicVoices` validation split
3. Define telephony degradation pipeline:
   - Resample to 8kHz
   - Butter bandpass 300–3400Hz
   - μ-law compress/expand
   - Add Gaussian noise (σ=0.002)
   - Simulate 2% packet loss (zero 20ms frames)
   - Resample back to 16kHz
4. Run clean accuracy evaluation
5. Run degraded accuracy evaluation
6. Run latency benchmark (50 inferences, wall-clock)
7. Print results table:
   ```
   | Language | Clean Acc | Degraded Acc | p50 Latency | p95 Latency |
   ```
8. Exit code 1 if acceptance criteria missed (clean <90%, degraded <80%, p95 >100ms)

### ✅ `backend/tests/test_language_id.py`

Self-contained — local fixtures only, `unittest.mock.patch` for model.

| # | Test | What it checks |
|---|---|---|
| 1 | `test_voxlingua_to_code_maps_all_five_languages` | Dict has all 5 mappings, no typos |
| 2 | `test_identify_language_fallback_on_unrecognised_label` | Unknown label → `"hi"` fallback |
| 3 | `test_identify_language_returns_confidence_from_model` | Confidence value propagated correctly |
| 4 | `test_identify_language_handles_code_colon_name_format` | `"hi: Hindi"` format parsed correctly |
| 5 | `test_update_active_language_switches_on_high_confidence_and_word_count` | ≥0.80 + >5 → switch |
| 6 | `test_update_active_language_stays_on_low_confidence` | <0.80 → no switch |
| 7 | `test_update_active_language_first_utterance_defaults_to_hi` | First + low conf → `"hi"` |
| 8 | `test_update_active_language_ignores_switch_on_low_word_count` | High conf but ≤5 words → no switch |
| 9 | `test_update_active_language_keeps_current_when_no_condition_met` | Nothing satisfied → current stays |

---

# Layer 2 — Voice Activity Detection — ✅ DONE

## Files created

### ✅ `backend/core/vad.py`

Module-level model loader + per-call `VADProcessor` class.

**`_load_model()`** — lazy-loads Silero VAD model via `silero_vad.load_silero_vad(onnx=True)`. The package manages ONNX session, GRU state, and context window internally. Falls back gracefully if unavailable.

**`VADProcessor`** — per-call state machine:
- `process(frame: np.ndarray, is_tts_active: bool = False) -> str | None`
- Events: `VAD_START` (speech just started), `VAD_END` (700ms consecutive silence after speech), `BARGE_IN` (speech during TTS playback)
- Circular pre-roll buffer of ~300ms (19 frames of 512 samples at 16kHz)
- `reset()` — clears all state + calls `_model.reset_states()`
- `get_utterance_audio()` — returns concatenated utterance as float32 array

**Decision logic:**
- `speech_prob > 0.5` during silence → `VAD_START`, start buffering (prepend pre-roll)
- `speech_prob > 0.5` during TTS → `BARGE_IN`, flush buffer, start fresh
- `speech_prob <= 0.5` for ~44 consecutive frames → `VAD_END`

### ✅ `backend/tests/test_vad.py`

`unittest.mock.patch` for ONNX session. 12 test cases:

| # | Test | What it checks |
|---|---|---|
| 1 | `test_vad_start_on_speech_after_silence` | Silence → speech → VAD_START |
| 2 | `test_vad_end_after_silence_threshold` | Speech → 44 silence frames → VAD_END |
| 3 | `test_barge_in_during_tts` | Speech during TTS → BARGE_IN |
| 4 | `test_pre_roll_buffer_captured_on_vad_start` | 300ms pre-roll prepended to utterance |
| 5 | `test_utterance_accumulates_during_speech` | Multiple frames appended correctly |
| 6 | `test_get_utterance_audio_returns_concatenated` | Concatenated array shape/dtype |
| 7 | `test_get_utterance_audio_empty_when_no_speech` | Empty before speech starts |
| 8 | `test_reset_clears_all_state` | State, buffers, hidden state all zeroed |
| 9 | `test_short_burst_does_not_emit_early_end` | Single silence frame no premature end |
| 10 | `test_full_idle_speaking_idle_cycle` | Complete cycle: idle → speaking → idle |
| 11 | `test_raises_on_wrong_frame_size` | ValueError on non-512-sample frame |
| 12 | `test_barge_in_flushes_previous_utterance` | Pre-roll preserved on barge-in |

## Verification commands

```bash
# All 29 tests pass (12 VAD + 17 existing)
uv run pytest backend/tests/ -v

# Formatting check
uv run black --check backend/core/vad.py backend/tests/test_vad.py
```
