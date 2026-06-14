# Custom Language ID — Small ECAPA-TDNN for 5 Indic Languages

**Date:** 2026-06-14  
**Status:** Design (pre-implementation)  
**Supersedes:** `docs/design.md` §5 (VoxLingua107 placeholder)

---

## Overview

Replace the pre-trained SpeechBrain VoxLingua107 ECAPA-TDNN with a purpose-built, smaller ECAPA-TDNN trained from scratch on our target 5 languages: Hindi (`hi`), English (`en`), Tamil (`ta`), Bengali (`bn`), Marathi (`mr`).

**Why:** VoxLingua107 is a 107-language classifier — most of its capacity is wasted on languages we don't need. A smaller model trained on only our 5 languages will be faster, use less RAM, and can be tuned to our specific audio domain (telephony-degraded).

---

## Data Pipeline

### Sources

| Language | Primary source | Fallback / supplement |
|---|---|---|
| `hi` | IndicVoices | — |
| `en` | Svarah (filtered) | — |
| `ta` | IndicVoices | — |
| `bn` | IndicVoices | — |
| `mr` | IndicVoices | — |

### Processing steps

1. **Stream from HuggingFace** — use `datasets` library, stream mode to avoid disk blowup.
2. **Filter** by language tag:
   - IndicVoices: keep `hi`, `ta`, `bn`, `mr` (Tamil is present as `ta` in IndicVoices).
   - Svarah: keep `en` only. Discard any samples with missing audio, zero-length, or SNR < 10dB.
3. **Split** 80/10/10 train/val/test per language. Stratified by speaker where speaker metadata exists.
4. **Resample** all audio to 16kHz mono, float32 in [-1, 1].
5. **Cache** to disk (`~/.cache/lang_id_dataset/`) so rebuilds are fast.

### Data augmentation (telephony simulation)

Apply to **60% of training samples** (random selection per epoch). Never to val or test.

- Resample 16kHz → 8kHz
- Bandpass filter (300–3400 Hz — PSTN passband)
- μ-law encode/decode (G.711 quantisation)
- Additive Gaussian noise (σ = 0.002)
- 2% packet loss simulation (zero random 20ms frames)
- Resample back 8kHz → 16kHz

Implementation: `training/augment.py`

---

## Model Architecture

Small ECAPA-TDNN for 5-class classification.

### Feature frontend
- 80-dimensional log-mel spectrograms
- 25ms window, 10ms shift, 64 FFT bins (n_fft=512)
- Computed on-the-fly in training, pre-computed in ONNX export

### Network

```
Input: (batch, 80, T) log-mel
  │
  ├─ Conv1D (80 → 512, kernel=5, stride=1, padding=2)
  ├─ BatchNorm + ReLU
  │
  ├─ SE-Res2Net block × 3 (C=512, scale=8, dilation=[2,3,4])
  ├─ SE-Res2Net block × 3 (C=512, scale=8, dilation=[2,3,4])
  ├─ SE-Res2Net block × 3 (C=512, scale=8, dilation=[2,3,4])
  │
  ├─ Attentive Statistics Pooling → (batch, 1024)
  ├─ BatchNorm
  ├─ FC (1024 → 512) + ReLU
  ├─ BatchNorm
  ├─ FC (512 → 5)
  └─ Softmax
```

- **Parameter count:** ~6M (VoxLingua107 original uses C=1024 with ~20M params)
- **Loss:** Cross-entropy with label smoothing (ε = 0.1)
- **Weights:** Saved as PyTorch `state_dict`, exported to ONNX fp32

### Why ECAPA-TDNN?
- State-of-the-art for speaker recognition and language ID
- Attentive pooling captures variable-length utterances well
- SE-Res2Net blocks are parameter-efficient
- Well-understood export path to ONNX

---

## Training Pipeline

### Setup
- **Framework:** PyTorch 2.x
- **Hardware:** Single GPU preferred (T4/RTX 3060+), single CPU fallback (slower but possible)
- **Batch size:** 32–64 (target: fits in ~4GB VRAM)

### Hyperparameters
| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Scheduler | Cosine decay + 5 epoch linear warmup |
| Epochs | 50 (early stopping patience=7 on val loss) |
| Label smoothing | 0.1 |
| Mixup (optional) | 0.2 alpha, applied before augmentation |

### Training script: `training/train_ecapa.py`

Runs:
1. Load cached dataset
2. Apply augmentation on-the-fly to 60% of train batch
3. Log per-epoch: train loss, val loss, per-language accuracy, confusion matrix
4. Save checkpoint at best val accuracy
5. Export best checkpoint to ONNX

### Validation
- Every epoch on clean (non-augmented) validation set
- Track: accuracy per language, macro F1, confusion matrix
- Monitor bn/mr confusion specifically (known hard pair)

---

## Model Export

`training/export_onnx.py`

- Load best PyTorch checkpoint
- Export to ONNX opset 18 with `torch.onnx.export()`
- Dynamic first dimension (batch) and dynamic T (time)
- Input: `(batch, 80, T)` float32 — log-mel spectrogram
- Output: `(batch, 5)` float32 — logits
- Validate: ONNX output matches PyTorch output within 1e-4 absolute diff (5 random test samples)
- Output: `backend/artifacts/lang_id_ecapa_ta.onnx`

---

## Inference Integration

### Replace `backend/core/language_id.py`

Remove:
- `speechbrain` import and `EncoderClassifier`
- `VOXLINGUA_TO_CODE` mapping
- `_load_classifier()` → `_load_model()` (ONNX)

Keep:
- `identify_language(audio, sr) → (lang_code, confidence)` function signature
- `update_active_language()` switching logic (no change needed)

New inference flow:

```python
import onnxruntime
import numpy as np

LANG_CODES = ["hi", "en", "ta", "bn", "mr"]
_model = None

def _load_model():
    session = onnxruntime.InferenceSession("backend/artifacts/lang_id_ecapa_ta.onnx")
    # Warm-up inference
    dummy = np.zeros((1, 80, 100), dtype=np.float32)
    session.run(None, {"mel": dummy})
    return session

def _log_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    """80-dim log-mel spectrogram. Pure numpy implementation."""
    # Uses librosa.stft / mel filterbank or hand-rolled STFT
    ...

def identify_language(audio: np.ndarray, sr: int = 16000):
    global _model
    if _model is None:
        _model = _load_model()
    mel = _log_mel(audio, sr)  # (1, 80, T)
    logits = _model.run(None, {"mel": mel})[0]  # (1, 5)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    idx = int(np.argmax(probs))
    confidence = float(probs[0][idx])
    lang_code = LANG_CODES[idx]
    return lang_code, confidence
```

### ONNX runtime considerations
- `onnxruntime` already in requirements (used by Moonshine STT models)
- No new inference dependency
- Log-mel computation uses librosa (small, pure-numpy subset or full install)

---

## File Structure

```
backend/
├── artifacts/
│   └── lang_id_ecapa_ta.onnx              # Trained model — gitignored
├── core/
│   └── language_id.py                      # Rewritten: ONNX + log-mel
├── training/
│   ├── __init__.py
│   ├── build_dataset.py                    # Stream, filter, cache, split
│   ├── augment.py                          # Telephony augmentation pipeline
│   ├── train_ecapa.py                      # PyTorch training loop
│   ├── export_onnx.py                      # Export + validate
│   └── eval_lang_id.py                     # Standalone eval on test set
├── tests/
│   └── test_language_id.py                 # Updated for new implementation
├── requirements.txt                        # Remove speechbrain
└── .gitignore                              # Add backend/artifacts/
```

`training/` is **dev-only**. Not included in Docker image or deployment.

---

## Dependencies

| Package | Status | Notes |
|---|---|---|
| `speechbrain` | **Remove** | No longer needed |
| `onnxruntime` | Keep | Already required for STT |
| `librosa` | **Add** (~5MB) | For log-mel extraction in inference |
| `torch` | Not in prod | Training/dev only |
| `datasets` | Not in prod | Training/dev only |
| `soundfile` | Not in prod | Training/dev only |

---

## Evaluation Plan

### Pre-deployment benchmarks

| Test | Dataset | Target |
|---|---|---|
| Clean accuracy | 50 utterances/lang from test split | ≥ 90% per language |
| Telephony-degraded accuracy | Same 250 utterances + augmentation | ≥ 80% per language |
| bn/mr confusion rate | Test set subset | < 15% |
| p95 CPU inference | 50 timed runs on Render-equivalent | < 50ms |
| Model load time | Cold start | < 2s |
| Peak RAM at inference | `memory_profiler` | < 60MB |

Run `training/eval_lang_id.py` after training, report numbers.

---

## Side Task: `te` → `ta` Rename

The original design doc incorrectly mapped Tamil to the code `te` (which is Telugu). This needs a **global rename** across the entire codebase:

| Location | Change |
|---|---|
| `docs/design.md` | "Tamil (te)" → "Tamil (ta)" |
| `backend/core/language_id.py` | `"te"` → `"ta"` in VOXLINGUA_TO_CODE (to be removed anyway) |
| `backend/core/stt.py` | Model key `"te"` → `"ta"` |
| `backend/core/tts.py` | Language mapping `"te"` → `"ta"` |
| `backend/schemas/appointment.py` | Any reference to `te` |
| `backend/db/migrations/001_create_tables.sql` | Comments/defaults |
| `docs/features.json` | Language list |
| All tests | `"te"` → `"ta"` test cases |

This is mechanical but must be done carefully to avoid missing references. (For the new custom lang ID, `"ta"` will be correct from the start.)
