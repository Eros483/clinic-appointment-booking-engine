# Custom ECAPA-TDNN Training Configuration

Trained on Kaggle (Tesla T4, ~124 min, 45 epochs).

## Task
5-language classification: hi, en, ta, bn, mr.

## Dataset
- **hi/ta/bn/mr** — `ai4bharat/IndicVoices` (train split, streaming)
- **en** — `ai4bharat/Svarah` (test split, streaming — only split available)
- **2000 samples/lang** → 1600 train / 200 val / 200 test per language
- Shuffled with `buffer_size=512, seed=42`
- Min duration: 0.5s; resampled to 16kHz mono

## Audio Preprocessing
| Param | Value |
|---|---|
| Sample rate | 16000 Hz |
| Window | 25ms (400 samples), Hanning |
| Hop | 10ms (160 samples) |
| FFT | 512 points |
| Mel bands | 80 |
| f_min | 0.0 |
| Transform | `torchaudio.transforms.MelSpectrogram` → `log(clamp(min=1e-10))` |

## Augmentation (60% probability per sample)
- Resample to 8kHz → bandpass 300–3400Hz (4th-order Butterworth)
- μ-law compand (mu=255)
- Additive Gaussian noise (σ=0.002)
- 2% random 20ms frame dropout (packet-loss simulation)

## Architecture: ECAPA-TDNN (C=512)
| Layer | Details |
|---|---|
| Conv1 | 80 → 512, kernel 5, padding 2, BN + ReLU |
| SE-Res2Block ×3 | dilation 2, 3, 4; scale 8; channels 512 |
| Pooling | Attentive Statistics Pooling (mean + std) → 1024 |
| FC | 1024 → 512, BN + ReLU |
| Classifier | 512 → 5 (no softmax) |

## Training
| Param | Value |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Batch size | 32 (variable-length padding collation) |
| Epochs | 45 |
| Warmup | Linear 0 → 1e-3 over 5 epochs |
| Schedule | Cosine decay (rest of 40 epochs) |
| Label smoothing | 0.1 |
| Early stopping | Patience 7 (val accuracy) |
| Mixed precision | No (model is small, ~6M params) |

## Results
| Metric | Value |
|---|---|
| Best val accuracy | 87.1% (epoch 44) |
| Test accuracy | 86.8% |
| Per-language test | hi 92%, en 87%, ta 80.5%, bn 94%, mr 80.5% |
| p50 latency | 4.2ms (T4 GPU, batch=32) |
| p95 latency | 6.0ms |
| ONNX diff | 4.77e-07 |
| PT vs ONNX disagreements | 0 / 1000 |

## Export
- Opset 18, legacy exporter (`dynamo=False`)
- Dynamic axes: batch (0) and time (2)
- Validated with onnxruntime on CPU
- Files: `lang_id_ecapa_ta.onnx` (146KB) + `lang_id_ecapa_ta.onnx.data` (11MB)
