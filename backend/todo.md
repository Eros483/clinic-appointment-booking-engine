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

## Verification commands

```bash
# Unit tests (9/9 passing)
uv run pytest backend/tests/test_language_id.py -v

# Formatting check (all 3 files clean)
uv run black --check backend/core/language_id.py backend/tests/test_language_id.py backend/scripts/eval_lang_id.py

# Pre-deployment validation on real model (requires ~2GB download + ~5min runtime)
# Downloads VoxLingua107 + IndicVoices dataset on first run
uv run python -m backend.scripts.eval_lang_id
```
