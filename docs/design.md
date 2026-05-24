# Clinic appointment booking engine — system spec v2

**Project type:** Resume / portfolio  
**Target:** Tier-1 Indian cities (Mumbai, Delhi, Bengaluru, Chennai, Kolkata)  
**Languages:** Hindi (`hi`), English (`en`), Tamil (`te`), Bengali (`bn`), Marathi (`mr`)

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Architecture layers](#2-architecture-layers)
3. [Layer 1 — Telephony](#3-layer-1--telephony)
4. [Layer 2 — Voice activity detection](#4-layer-2--voice-activity-detection-vad)
5. [Layer 3 — Language identification](#5-layer-3--language-identification)
6. [Layer 4 — Speech-to-text](#6-layer-4--speech-to-text-stt)
7. [Layer 5 — LLM orchestration](#7-layer-5--llm-orchestration)
8. [Layer 6 — Text-to-speech](#8-layer-6--text-to-speech-tts)
9. [Training data & augmentation](#9-training-data--telephony-augmentation)
10. [Call state machine](#10-call-state-machine)
11. [Data layer — Postgres schema](#11-data-layer--postgres-schema)
12. [Session state — Redis](#12-session-state--redis)
13. [Observability — LangSmith](#13-observability--langsmith)
14. [Metrics & evaluation](#14-metrics--evaluation)
15. [Admin dashboard](#15-admin-dashboard)
16. [Deployment](#16-deployment)
17. [Tech stack summary](#17-tech-stack-summary)
18. [Open questions & future work](#18-open-questions--future-work)

---

## 1. System overview

A real-time, multilingual voice agent that answers a clinic's phone line, identifies the caller's language, conducts a natural conversation to collect appointment details, validates the slot against live doctor availability in Postgres, and autonomously schedules the event on the doctor's Google Calendar while sending the patient a confirmation email — all without human intervention.

```
Caller dials number
     │
     ▼
Twilio (PSTN → WebSocket audio stream)
     │  8kHz μ-law G.711
     ▼
VAD (Silero, per-frame)
     │  end-of-utterance event
     ▼
Language classifier (SpeechBrain VoxLingua107 ECAPA-TDNN, 5-class subset)
     │  active_language
     ▼
STT router → Moonshine Tiny ONNX Q8 [hi|en|te|bn|mr]
     │  transcript text
     ▼
LLM orchestrator (Groq) ←→ LangSmith tracing
     │  response text + tool calls
     ├──► Postgres slot availability check (real-time, during confirmation)
     ├──► Google Calendar MCP (post-call async)
     ├──► Email MCP (post-call async)
     └──► Postgres write (real-time)
     │
     ▼
TTS — Sarvam Bulbul v3 (language-matched voice)
     │  audio stream
     ▼
Twilio → caller
```

---

## 2. Architecture layers

| # | Layer | Technology | Runs |
|---|---|---|---|
| 1 | Telephony | Twilio Media Streams | Cloud (free tier) |
| 2 | VAD | Silero VAD (ONNX) | Server (per-frame) |
| 3 | Language ID | SpeechBrain VoxLingua107 ECAPA-TDNN | Server (per utterance) |
| 4 | STT | Moonshine Tiny ONNX Q8 ×5 (warm-loaded) | Server (per utterance) |
| 5 | Orchestration | Groq inference API (Llama 3.1 8B) | API (free tier) |
| 6 | TTS | Sarvam Bulbul v3 | API (free tier) |
| 7 | Slot check | Postgres `doctor_slots` table | Server (real-time) |
| 8 | Calendar | Google Calendar MCP | Post-call async |
| 9 | Email | Email provider MCP | Post-call async |
| 10 | Observability | LangSmith | API (free tier) |
| 11 | Session state | Redis | Server |
| 12 | Persistence | Postgres | Render free tier |
| 13 | Dashboard | React (Vercel) + FastAPI | Cloud |

---

## 3. Layer 1 — Telephony

### Twilio setup

- Inbound PSTN number → Twilio webhook triggers a TwiML response that opens a **Media Stream** WebSocket.
- Audio format: **8kHz, μ-law (G.711), mono**, delivered in 20ms frames (~160 bytes per chunk).
- The server receives frames over the WebSocket, decodes μ-law, and **resamples 8kHz → 16kHz** before passing to VAD and downstream models.

```python
import audioop
import numpy as np
from scipy.signal import resample_poly

def decode_twilio_frame(mulaw_bytes: bytes) -> np.ndarray:
    pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)           # μ-law → 16-bit PCM
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_16k = resample_poly(pcm, up=2, down=1)           # 8kHz → 16kHz
    return audio_16k
```

> **Note:** Use `resample_poly` (integer ratio, polyphase filter) rather than `librosa.resample` for per-chunk real-time use. It is significantly faster and avoids FFT overhead.

### Barge-in: killing TTS output

When the caller speaks over the bot, the ongoing TTS stream must be halted immediately. The VAD barge-in event must:

1. Kill the TTS audio queue.
2. Send a Twilio `<Stop>` command via the media stream.
3. Flush the utterance buffer and begin collecting the new speech.

### Free-tier constraints

Twilio free tier requires a verified caller ID for outbound; inbound calls work without restriction. The trial number will announce a Twilio promotional message on first connection — acceptable for a portfolio demo. Document this in the README.

---

## 4. Layer 2 — Voice activity detection (VAD)

**Model:** [Silero VAD](https://github.com/snakers4/silero-vad)  
**Input:** 16kHz PCM frames (512 samples = 32ms window)

### Decision logic

```
Silero VAD (per frame, 512-sample window)
 ├─ speech_prob > 0.5, during silence   → start buffering utterance
 ├─ speech_prob > 0.5, during TTS out   → BARGE-IN: kill TTS, flush buffer, start buffering
 └─ 700ms consecutive silence after speech → end of utterance → trigger pipeline
```

### Implementation notes

- Run Silero in ONNX for minimal overhead; fits comfortably in <5ms per frame on a single CPU core.
- 700ms end-of-utterance threshold is a starting point. Tune to 500ms if conversation feels sluggish.
- Keep a **circular pre-roll buffer** of ~300ms so that the first 300ms of speech before the VAD trigger is captured.

---

## 5. Layer 3 — Language identification

### Model: SpeechBrain VoxLingua107 ECAPA-TDNN

Rather than training a custom classifier, we use the pre-trained [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) checkpoint from HuggingFace. This model was trained on 107 languages including all five target languages (Hindi, English, Tamil, Bengali, Marathi).

**Before deploying**, the model must be validated on IndicVoices to confirm it meets accuracy and latency requirements at telephony audio quality. This is a required pre-build step, not an afterthought.

---

### Pre-deployment validation on IndicVoices

#### Step 1 — Build a telephony-degraded test set

Sample 50 utterances per language (250 total) from the IndicVoices validation split. Apply the full telephony augmentation pipeline (see §9) to simulate Twilio's 8kHz μ-law audio. Do **not** apply augmentation to the test set used for the final accuracy number — run two evaluations: one on clean audio and one on degraded audio to quantify the domain gap.

#### Step 2 — Accuracy benchmark

```python
import torch
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp/lang-id"
)

LANG_MAP = {
    "hi": "Hindi",  # VoxLingua107 label
    "en": "English",
    "te": "Tamil",
    "bn": "Bengali",
    "mr": "Marathi",
}

TARGET_LANGS = set(LANG_MAP.values())

def classify(audio_path: str) -> tuple[str, float]:
    prediction = classifier.classify_file(audio_path)
    label = prediction[3][0]          # top predicted label
    confidence = prediction[1][0].item()
    return label, confidence

# Run across 250 test utterances, compute per-language accuracy and confusion matrix
```

#### Step 3 — Latency benchmark

Measure inference wall-clock time on the target Render instance (or equivalent CPU). The target is **<100ms per utterance** so the language ID step does not dominate end-to-end turn latency. Run 50 timed inferences and report p50 and p95.

```python
import time

latencies = []
for audio_path in test_files:
    t0 = time.perf_counter()
    classify(audio_path)
    latencies.append((time.perf_counter() - t0) * 1000)

print(f"p50: {sorted(latencies)[len(latencies)//2]:.1f}ms")
print(f"p95: {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
```

#### Acceptance criteria

| Metric | Target | Action if missed |
|---|---|---|
| Accuracy (clean audio) | ≥ 90% per language | Document gap; consider fallback to Whisper `detect_language()` for that language |
| Accuracy (telephony-degraded) | ≥ 80% per language | Adjust confidence threshold downward; log more aggressively |
| p95 latency (CPU) | < 100ms | Run model in ONNX export via `speechbrain`'s export utilities |
| bn/mr confusion rate | < 15% | Flag in README; increase confidence threshold for switching |

> **Resume note:** "Benchmarked `speechbrain/lang-id-voxlingua107-ecapa` on telephony-augmented IndicVoices across 5 Indic languages; achieved X% avg accuracy at p95 latency of Yms on CPU" is a credible, verifiable claim. Run the eval. Log the numbers.

---

### Inference integration

```python
from speechbrain.pretrained import EncoderClassifier
import numpy as np

_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp/lang-id"
)

VOXLINGUA_TO_CODE = {
    "Hindi": "hi", "English": "en", "Tamil": "te",
    "Bengali": "bn", "Marathi": "mr",
}

def identify_language(audio: np.ndarray, sr: int = 16000) -> tuple[str, float]:
    """Returns (lang_code, confidence). Falls back to 'hi' on unrecognised label."""
    signal = torch.tensor(audio).unsqueeze(0)
    prediction = _classifier.classify_batch(signal)
    label = prediction[3][0]
    confidence = prediction[1][0].item()
    lang_code = VOXLINGUA_TO_CODE.get(label, "hi")
    return lang_code, confidence
```

### Switching logic

```python
def update_active_language(
    prediction: str,
    confidence: float,
    word_count: int,
    current_language: str,
    is_first_utterance: bool,
) -> str:
    if confidence >= 0.80 and word_count > 5:
        return prediction
    if is_first_utterance and confidence < 0.80:
        return "hi"           # default to Hindi on low-confidence first utterance
    return current_language   # keep previous
```

---

## 6. Layer 4 — Speech-to-text (STT)

**Model:** [Moonshine Tiny](https://github.com/usefulsensors/moonshine) exported to **ONNX INT8 (Q8)** quantization, one model per language.

### Deployment

- All five language models **pre-loaded into RAM** at startup (warm starts only).
- Estimated memory: ~40–60MB per Q8 model → ~250–300MB total. Render free tier provides 512MB RAM — monitor headroom carefully alongside the SpeechBrain classifier and Redis.
- Route each utterance buffer to the model matching `active_language`.

```python
STT_MODELS: dict[str, onnxruntime.InferenceSession] = {
    "hi": load_onnx("moonshine-tiny-hi-q8.onnx"),
    "en": load_onnx("moonshine-tiny-en-q8.onnx"),
    "te": load_onnx("moonshine-tiny-te-q8.onnx"),
    "bn": load_onnx("moonshine-tiny-bn-q8.onnx"),
    "mr": load_onnx("moonshine-tiny-mr-q8.onnx"),
}

def transcribe(audio: np.ndarray, language: str) -> str:
    session = STT_MODELS[language]
    # ... run inference
```

> **Render free-tier RAM note:** If 512MB is too tight with all models warm-loaded simultaneously, lazy-load the two least-used languages and accept a one-time cold-load latency for those. Log model-load events so you can see if it's actually a problem in practice.

---

## 7. Layer 5 — LLM orchestration

### Model

**Groq inference API** — Llama 3.1 8B (fast, free tier).

> **Groq free-tier limits:** As of late 2024, the free tier allows 30 requests/minute and 14,400 requests/day per model. Document these in the README. Add a queued retry with exponential backoff and a canned "one moment please" TTS message on rate-limit errors.

### Responsibilities

1. **Clinic verification** — confirm caller reached the correct clinic on first utterance.
2. **Appointment slot extraction** — extract structured JSON:
   ```json
   {
     "doctor_name": "Dr. Meera Joshi",
     "preferred_date": "2024-12-15",
     "preferred_time": "10:30",
     "patient_name": "Ramesh Kumar",
     "contact_number": "+919876543210",
     "complaint": "knee pain follow-up"
   }
   ```
3. **Slot availability check** — before entering `CONFIRM_SLOT` state, query Postgres `doctor_slots` to verify the requested slot is free (see §11).
4. **Confirmation loop** — read extracted details back to the caller including confirmed slot; ask for explicit confirmation.
5. **Post-call tool calls (async):** Google Calendar MCP and Email MCP.

### Prompt strategy

- System prompt sets the persona, lists required JSON fields, and instructs the model to respond in `active_language`.
- Use XML tags to delimit extraction: `<appointment_data>...</appointment_data>`.
- Full conversation history sent on every turn (LLM is stateless).

### Validation layer (post-extraction)

```python
from datetime import datetime, timedelta

def validate_appointment(data: dict) -> list[str]:
    errors = []
    try:
        dt = datetime.strptime(data["preferred_date"], "%Y-%m-%d")
        if dt < datetime.now() or dt > datetime.now() + timedelta(days=90):
            errors.append("Date out of valid booking window")
    except (ValueError, KeyError):
        errors.append("Invalid or missing date")
    if not data.get("doctor_name"):
        errors.append("Doctor name missing")
    return errors
```

### LangSmith integration

All LLM calls are wrapped with LangSmith tracing. See §13 for full setup.

### Graceful degradation on Groq rate limit

```python
import asyncio

async def call_groq_with_retry(messages: list, max_retries: int = 2) -> str:
    for attempt in range(max_retries):
        try:
            return await groq_client.chat(messages)
        except RateLimitError:
            if attempt == 0:
                await play_tts("one moment please", lang=session["active_language"])
            await asyncio.sleep(2 ** attempt)
    raise RuntimeError("Groq rate limit exceeded after retries")
```

---

## 8. Layer 6 — Text-to-speech (TTS)

**Model:** [Sarvam Bulbul v3](https://www.sarvam.ai/) — free tier.

> **Sarvam free-tier note:** Check current rate limits on the Sarvam dashboard before demo. Cache common phrases (greeting, "one moment please", confirmation template) as pre-generated audio bytes at startup to avoid API calls on critical path utterances.

### Language-to-voice mapping

```python
LANGUAGE_TO_BULBUL = {
    "hi": {"target_language_code": "hi-IN", "speaker": "anand"},
    "en": {"target_language_code": "en-IN", "speaker": "anand"},
    "te": {"target_language_code": "te-IN", "speaker": "anu"},
    "bn": {"target_language_code": "bn-IN", "speaker": "anu"},
    "mr": {"target_language_code": "mr-IN", "speaker": "anu"},
}

def get_tts_params(lang: str, confidence: float, word_count: int) -> dict | None:
    if confidence < 0.80 or word_count <= 5:
        return None      # insufficient confidence — keep current voice
    return LANGUAGE_TO_BULBUL.get(lang, LANGUAGE_TO_BULBUL["hi"])
```

Twilio expects 8kHz μ-law for playback; resample Bulbul's output (22–24kHz PCM) back to 8kHz before sending.

---

## 9. Training data & telephony augmentation

### Dataset

**[ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices)** — used for language classifier validation and STT fine-tuning.

IndicVoices is recorded at 16–48kHz with high SNR. Twilio delivers 8kHz μ-law band-limited to 300Hz–3.4kHz with compression artifacts. Without domain adaptation, WER and classifier accuracy drops of 30–50% are common in production.

### Augmentation pipeline

Apply to **60% of training samples** (never validation or test sets):

```python
import librosa
import numpy as np
from scipy.signal import butter, sosfilt

def simulate_telephony(audio: np.ndarray, sr: int) -> np.ndarray:
    # 1. Resample to 8kHz
    audio_8k = librosa.resample(audio, orig_sr=sr, target_sr=8000)

    # 2. Bandpass filter (300Hz–3400Hz PSTN passband)
    sos = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")
    audio_8k = sosfilt(sos, audio_8k)

    # 3. μ-law encode/decode (G.711 quantisation artefacts)
    audio_mulaw = librosa.mu_compress(audio_8k, mu=255, quantize=True)
    audio_decoded = librosa.mu_expand(audio_mulaw, mu=255, quantize=True)

    # 4. Additive background noise
    noise = np.random.normal(0, 0.002, audio_decoded.shape)
    audio_noisy = audio_decoded + noise

    # 5. Simulate packet loss (~2% of 20ms frames zeroed)
    frame_len = int(8000 * 0.02)
    for i in range(0, len(audio_noisy) - frame_len, frame_len):
        if np.random.rand() < 0.02:
            audio_noisy[i:i + frame_len] = 0.0

    # 6. Resample back to 16kHz for model input
    audio_16k = librosa.resample(audio_noisy, orig_sr=8000, target_sr=16000)
    return audio_16k
```

### Augmentation split

| Subset | Augmented | Reason |
|---|---|---|
| Training | 60% | Domain adaptation without overfitting to telephony artifacts |
| Validation | 0% | Clean audio → representative of held-out real quality |
| Test | 0% | Clean audio → unbiased benchmark |

---

## 10. Call state machine

An explicit state machine prevents the LLM from being the sole source of conversation position. The slot availability check is a mandatory gate before `CONFIRM_SLOT`. State is stored in Redis keyed by `call_sid`.

```
        ┌────────────┐
 ──────►│  GREETING  │
        └─────┬──────┘
              │ clinic confirmed
        ┌─────▼──────────┐
        │ COLLECT_DETAILS│◄──────────────┐
        └─────┬──────────┘               │ missing fields
              │ all fields extracted     │
        ┌─────▼──────────┐               │
        │  CHECK_SLOT    │               │ slot taken → ask for alternative
        └─────┬──────────┘               │
              │ slot available ──────────►│ (re-enter COLLECT_DETAILS with date/time cleared)
        ┌─────▼──────────┐
        │  CONFIRM_SLOT  │
        └─────┬──────────┘
              │ caller confirms
        ┌─────▼──────────┐
        │  COMMITTING    │  (write to Postgres, trigger MCP tool calls)
        └─────┬──────────┘
              │
        ┌─────▼──────────┐
        │     DONE       │  (call ends, Redis flushed to Postgres)
        └────────────────┘

 Special transitions:
   Any state + wrong clinic confirmed → WRONG_CLINIC (terminate + log)
   Any state + timeout (30s silence)  → ABANDONED    (terminate + log)
   CHECK_SLOT + slot taken            → COLLECT_DETAILS (clear date/time, prompt for alternative)
```

### Redis session schema

```json
{
  "call_sid": "CAxxxxxxxx",
  "state": "COLLECT_DETAILS",
  "active_language": "hi",
  "language_confidence": 0.93,
  "turn_count": 4,
  "timings": {
    "call_started_at": "2024-12-01T10:30:00Z",
    "last_vad_end": 1733048600123,
    "last_llm_first_token": 1733048600543,
    "last_turn_complete": 1733048601100
  },
  "partial_appointment": {
    "doctor_name": "Dr. Sharma",
    "preferred_date": null,
    "preferred_time": null,
    "patient_name": "Priya",
    "contact_number": null,
    "complaint": "fever"
  },
  "conversation_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "started_at": "2024-12-01T10:30:00Z"
}
```

TTL: 3600 seconds.

---

## 11. Data layer — Postgres schema

### `doctor_slots` ← new table

Stores available and booked slots per doctor. The slot availability check in `CHECK_SLOT` state queries this table directly — no live Calendar read required during a call.

| Column | Type | Notes |
|---|---|---|
| `slot_id` | `UUID PK` | |
| `doctor_name` | `TEXT` | Must match names used in `appointments` |
| `slot_datetime` | `TIMESTAMPTZ` | Slot start time |
| `duration_minutes` | `INT` | Default 15 or 30 |
| `status` | `TEXT` | `available`, `booked`, `blocked` |
| `appointment_id` | `UUID FK → appointments` | Null if `available` |
| `created_at` | `TIMESTAMPTZ` | |

#### Slot availability check

Called during `CHECK_SLOT` state, before presenting the slot to the caller for confirmation:

```python
import asyncpg
from datetime import datetime

async def is_slot_available(
    conn: asyncpg.Connection,
    doctor_name: str,
    slot_datetime: datetime,
) -> bool:
    row = await conn.fetchrow(
        """
        SELECT status FROM doctor_slots
        WHERE doctor_name = $1
          AND slot_datetime = $2
        """,
        doctor_name,
        slot_datetime,
    )
    if row is None:
        return False        # slot not in schedule at all
    return row["status"] == "available"

async def book_slot(
    conn: asyncpg.Connection,
    doctor_name: str,
    slot_datetime: datetime,
    appointment_id: str,
) -> bool:
    """Atomically marks a slot as booked. Returns False if already taken (race condition guard)."""
    result = await conn.execute(
        """
        UPDATE doctor_slots
        SET status = 'booked', appointment_id = $3
        WHERE doctor_name = $1
          AND slot_datetime = $2
          AND status = 'available'
        """,
        doctor_name,
        slot_datetime,
        appointment_id,
    )
    return result == "UPDATE 1"
```

> **Race condition note:** The `WHERE status = 'available'` in the `UPDATE` is a lightweight optimistic lock. If two concurrent calls target the same slot, only one `UPDATE` will return `UPDATE 1`. The losing call should re-enter `COLLECT_DETAILS` and ask the caller to choose a different time.

#### Seeding the schedule

For the portfolio demo, pre-seed `doctor_slots` with a week of 15-minute slots per doctor via a migration script. This avoids the complexity of live calendar sync while keeping the slot-check logic fully real.

```sql
-- Example: generate slots for Dr. Sharma, Mon–Fri, 9am–5pm, 15-min intervals
INSERT INTO doctor_slots (slot_id, doctor_name, slot_datetime, duration_minutes, status)
SELECT
    gen_random_uuid(),
    'Dr. Sharma',
    generate_series(
        '2024-12-02 09:00:00+05:30'::timestamptz,
        '2024-12-06 17:00:00+05:30'::timestamptz,
        '15 minutes'::interval
    ),
    15,
    'available';
```

---

### `calls`

| Column | Type | Notes |
|---|---|---|
| `call_sid` | `TEXT PK` | Twilio call SID |
| `started_at` | `TIMESTAMPTZ` | |
| `ended_at` | `TIMESTAMPTZ` | |
| `duration_seconds` | `INT` | |
| `detected_language` | `TEXT` | Final `active_language` |
| `language_switches` | `INT` | |
| `status` | `TEXT` | `completed`, `wrong_number`, `abandoned`, `error` |
| `transcript_json` | `JSONB` | Full turn-by-turn transcript |
| `avg_turn_latency_ms` | `FLOAT` | Computed at call end from per-turn logs |
| `avg_ttft_ms` | `FLOAT` | Average time-to-first-token across all LLM calls |
| `created_at` | `TIMESTAMPTZ` | |

### `appointments`

| Column | Type | Notes |
|---|---|---|
| `appointment_id` | `UUID PK` | |
| `call_sid` | `TEXT FK → calls` | |
| `doctor_name` | `TEXT` | |
| `slot_datetime` | `TIMESTAMPTZ` | Confirmed slot |
| `patient_name` | `TEXT` | |
| `patient_phone_hash` | `TEXT` | SHA-256 of phone number |
| `complaint` | `TEXT` | |
| `calendar_event_id` | `TEXT` | Google Calendar event ID |
| `email_sent` | `BOOL` | |
| `created_at` | `TIMESTAMPTZ` | |

### `turn_metrics`

Per-turn timing log. Source of truth for the metrics in §14.

| Column | Type | Notes |
|---|---|---|
| `id` | `BIGSERIAL PK` | |
| `call_sid` | `TEXT FK → calls` | |
| `turn_index` | `INT` | 0-indexed turn number |
| `vad_end_ts` | `BIGINT` | Unix ms: VAD end-of-utterance event |
| `stt_complete_ts` | `BIGINT` | Unix ms: transcript ready |
| `llm_first_token_ts` | `BIGINT` | Unix ms: first token received from Groq |
| `tts_first_audio_ts` | `BIGINT` | Unix ms: first audio byte sent to Twilio |
| `ttft_ms` | `INT` | `llm_first_token_ts - stt_complete_ts` |
| `turn_latency_ms` | `INT` | `tts_first_audio_ts - vad_end_ts` |
| `active_language` | `TEXT` | Language at this turn |

### `language_events`

| Column | Type | Notes |
|---|---|---|
| `id` | `BIGSERIAL PK` | |
| `call_sid` | `TEXT FK → calls` | |
| `event_time` | `TIMESTAMPTZ` | |
| `from_language` | `TEXT` | |
| `to_language` | `TEXT` | |
| `confidence` | `FLOAT` | |
| `utterance_word_count` | `INT` | |

---

## 12. Session state — Redis

Redis handles ephemeral per-call state with sub-millisecond access during live calls. Postgres handles durable records after call end.

```
Key:    session:{call_sid}
Value:  JSON blob (see §10 schema)
TTL:    3600s
```

The `timings` block in the session JSON is updated in real-time during each turn and flushed to `turn_metrics` (Postgres) at call end.

On call end (`call.completed` webhook): flush Redis session → write `calls` and `turn_metrics` rows → delete Redis key.

---

## 13. Observability — LangSmith

LangSmith traces all LLM calls, capturing inputs, outputs, latency, and token usage. This is the primary tool for debugging conversation failures and monitoring prompt behaviour across the 20 evaluation calls.

### Setup

```python
import os
from langsmith import Client
from langsmith.wrappers import wrap_openai  # works with any OpenAI-compatible client

os.environ["LANGCHAIN_API_KEY"] = "<your-langsmith-key>"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "clinic-voice-agent"
```

### Wrapping the Groq client

Groq exposes an OpenAI-compatible API. Wrap it directly:

```python
from openai import AsyncOpenAI
from langsmith.wrappers import wrap_openai

groq_client = wrap_openai(
    AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
)
```

All calls through this client are automatically traced. No changes needed to the LLM call sites.

### Per-call trace context

Tag each trace with `call_sid` and `active_language` so you can filter traces by call in the LangSmith UI:

```python
from langsmith import traceable

@traceable(
    name="llm_turn",
    metadata={"call_sid": call_sid, "language": active_language, "turn": turn_index},
)
async def run_llm_turn(messages: list) -> str:
    response = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        stream=True,
    )
    # record llm_first_token_ts here, on first chunk received
    first_chunk = True
    result = ""
    async for chunk in response:
        if first_chunk:
            session["timings"]["last_llm_first_token"] = time.time_ns() // 1_000_000
            first_chunk = False
        result += chunk.choices[0].delta.content or ""
    return result
```

### What LangSmith gives you

| Signal | Where to see it |
|---|---|
| TTFT per LLM call | Trace timeline → streaming latency |
| Full prompt + response | Trace inputs/outputs |
| Token usage per call | Trace metadata |
| Rate limit errors | Trace errors tab |
| Prompt regressions across calls | Compare runs view |

> **LangSmith free tier:** 5,000 traces/month. At ~5–10 LLM calls per appointment call, this is sufficient for the 20-call evaluation and ongoing development.

---

## 14. Metrics & evaluation

Three primary metrics are measured across **20 scripted test calls** before any public demo or resume claim.

---

### Metric 1 — Time to First Token (TTFT)

**Definition:** Elapsed time from STT transcript ready → first token received from Groq.

**Why it matters:** TTFT is the dominant contributor to perceived latency during conversation. It is owned almost entirely by the Groq API and network round-trip — capturing it surfaces infrastructure problems, not application code problems.

**How to measure:**

```python
# In run_llm_turn() (see §13):
stt_complete_ts = session["timings"]["last_stt_complete"]   # set after transcribe()
llm_first_token_ts = session["timings"]["last_llm_first_token"]  # set on first chunk

ttft_ms = llm_first_token_ts - stt_complete_ts

# Write to turn_metrics at call end
```

**Target:** < 400ms (Groq is fast; typical observed TTFT is 150–300ms on Llama 3.1 8B)

**Report:** Average TTFT across all turns across 20 test calls.

---

### Metric 2 — Average End-to-End Turn Latency

**Definition:** Elapsed time from VAD end-of-utterance event → first audio byte sent to Twilio.

This covers the full pipeline: STT + language ID + LLM (streaming) + TTS first chunk + Twilio send.

**How to measure:**

```python
vad_end_ts = session["timings"]["last_vad_end"]         # set in VAD handler
tts_first_audio_ts = session["timings"]["last_tts_first_audio"]  # set when first audio chunk sent

turn_latency_ms = tts_first_audio_ts - vad_end_ts

# Write to turn_metrics at call end
```

**Target:** < 800ms end-to-end

**Breakdown budget (approximate):**

| Stage | Budget |
|---|---|
| STT (Moonshine Tiny Q8) | ~100ms |
| Language ID (VoxLingua107) | ~80ms |
| LLM TTFT (Groq) | ~300ms |
| TTS first chunk (Sarvam) | ~200ms |
| Network/overhead | ~100ms |
| **Total** | **~780ms** |

**Report:** Average and p95 turn latency across all turns across 20 test calls.

---

### Metric 3 — Language Classification Accuracy

**Definition:** Percentage of utterances where the classifier's `active_language` matches the ground-truth language spoken, across the 20 evaluation calls.

**How to measure:**

The 20 test calls are scripted with known languages. After each call, cross-reference `language_events` in Postgres against the ground-truth language log to count misclassifications.

```python
# Evaluation script (run post-call)
import asyncpg

async def compute_lang_accuracy(call_sids: list[str]) -> dict:
    conn = await asyncpg.connect(DATABASE_URL)
    
    results = {}
    for call_sid, ground_truth_lang in zip(call_sids, ground_truth_langs):
        events = await conn.fetch(
            "SELECT to_language, confidence FROM language_events WHERE call_sid = $1",
            call_sid,
        )
        # Count turns where predicted language == ground truth
        correct = sum(1 for e in events if e["to_language"] == ground_truth_lang)
        results[call_sid] = correct / len(events) if events else 0.0
    
    return results
```

**Test call design (20 calls across 5 languages):**

| Language | # of calls | Notes |
|---|---|---|
| Hindi | 5 | Most common; baseline |
| English | 4 | Mix of Indian-accented English |
| Tamil | 4 | Include 1 code-switching call (Tamil + English) |
| Bengali | 4 | Include 1 where bn/mr confusion is likely |
| Marathi | 3 | Focus on cases near Bengali boundary |

**Target:** ≥ 85% accuracy averaged across all 20 calls, ≥ 75% per language.

**Report:** Per-language accuracy + overall average. If a language misses the target, document it — a known, quantified gap is more credible than silence.

---

### Metrics summary table (fill in after running evals)

| Metric | Target | Measured |
|---|---|---|
| Avg TTFT | < 400ms | ___ ms |
| p95 TTFT | < 600ms | ___ ms |
| Avg turn latency | < 800ms | ___ ms |
| p95 turn latency | < 1200ms | ___ ms |
| Overall lang accuracy | ≥ 85% | ___ % |
| Per-language min accuracy | ≥ 75% | ___ % (lang: ___) |
| Booking success rate | ≥ 80% | ___ % |

This table goes directly into the README and forms the basis for resume bullet numbers.

---

## 15. Admin dashboard

Read-only interface for clinic staff and the developer. React frontend on Vercel, backed by a FastAPI REST API on the same Render instance as the bot.

### Views

| View | Data source | Shows |
|---|---|---|
| **Appointment list** | `appointments JOIN calls` | All booked slots, filterable by date / doctor |
| **Doctor availability** | `doctor_slots` | Slot grid per doctor, colour-coded by status |
| **Call log** | `calls` | Per-call status, duration, language, transcript viewer |
| **Metrics dashboard** | `turn_metrics`, `calls` | TTFT, turn latency histograms, language distribution |
| **Language analytics** | `language_events` | Switch frequency, confidence histograms |

### Stack

- **Backend:** FastAPI serving REST endpoints over Postgres (same Render service as the bot)
- **Frontend:** React on Vercel — keep it simple, no auth framework needed for a portfolio demo. One shared API key is fine.
- **CORS:** FastAPI `CORSMiddleware` configured to allow the Vercel domain.

### Key API endpoints

```
GET  /calls                  → paginated call list with status + latency summary
GET  /calls/{call_sid}       → full call record + transcript JSON
GET  /appointments           → appointment list, filter by ?doctor=&date=
GET  /slots/{doctor_name}    → available/booked slot grid
GET  /metrics/latency        → aggregated TTFT and turn latency stats
GET  /metrics/languages      → language distribution and switch events
```

---

## 16. Deployment

All services use free tiers. No credit card required for initial deployment. Document tier limits in the README so a recruiter understands the constraints.

### Service map

| Service | Provider | Free tier limits | Notes |
|---|---|---|---|
| Telephony | Twilio | Trial credit (~$15); inbound calls free with number | Trial number announces Twilio promo message |
| Bot server | Render (Web Service) | 512MB RAM, 0.1 CPU, spins down after 15min inactivity | Add a `/health` ping to keep warm |
| Postgres | Render (PostgreSQL) | 1GB storage, 97 days expiry | Export and re-create before expiry |
| Redis | Render (Redis) | 25MB | Sufficient for <10 concurrent calls |
| LLM | Groq | 30 req/min, 14,400 req/day | Add retry logic (see §7) |
| TTS | Sarvam | Check current dashboard limits | Cache common phrases at startup |
| Observability | LangSmith | 5,000 traces/month | Sufficient for dev + 20 eval calls |
| Frontend | Vercel | Unlimited deploys, 100GB bandwidth | Deploy React dashboard here |

### Docker setup (Render)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SpeechBrain model at build time (avoids cold-start download)
RUN python -c "from speechbrain.pretrained import EncoderClassifier; \
    EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='tmp/lang-id')"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# render.yaml
services:
  - type: web
    name: clinic-voice-agent
    runtime: docker
    plan: free
    envVars:
      - key: TWILIO_ACCOUNT_SID
        sync: false
      - key: TWILIO_AUTH_TOKEN
        sync: false
      - key: GROQ_API_KEY
        sync: false
      - key: SARVAM_API_KEY
        sync: false
      - key: LANGCHAIN_API_KEY
        sync: false
      - key: LANGCHAIN_TRACING_V2
        value: "true"
      - key: LANGCHAIN_PROJECT
        value: "clinic-voice-agent"
      - key: DATABASE_URL
        fromDatabase:
          name: clinic-db
          property: connectionString
      - key: REDIS_URL
        fromService:
          name: clinic-redis
          property: connectionString

databases:
  - name: clinic-db
    plan: free

  - name: clinic-redis
    type: redis
    plan: free
```

### Render spin-down handling

Render free tier spins down web services after 15 minutes of inactivity. A Twilio webhook hitting a spun-down service will time out. Two mitigations:

1. **UptimeRobot** (free) — ping `/health` every 5 minutes to keep the service warm.
2. Document the cold-start latency (~30s) in the README as a known free-tier constraint.

### Environment variables

Never commit secrets. Use Render's environment variable dashboard. Locally, use a `.env` file with `python-dotenv`.

```
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
GROQ_API_KEY=
SARVAM_API_KEY=
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=clinic-voice-agent
DATABASE_URL=
REDIS_URL=
GOOGLE_CALENDAR_CREDENTIALS_JSON=
```

### Vercel frontend deployment

```bash
# In the /dashboard directory
vercel --prod
```

Set `VITE_API_BASE_URL=https://clinic-voice-agent.onrender.com` in Vercel environment variables.

---

## 17. Tech stack summary

| Component | Technology | Tier |
|---|---|---|
| Telephony | Twilio Media Streams | Free (trial) |
| VAD | Silero VAD (ONNX) | Open source |
| Language ID | SpeechBrain VoxLingua107 ECAPA | Open source / HuggingFace |
| STT | Moonshine Tiny ONNX Q8 ×5 | Open source |
| LLM | Groq API (Llama 3.1 8B) | Free tier |
| TTS | Sarvam Bulbul v3 | Free tier |
| Observability | LangSmith | Free tier |
| Slot check | Postgres `doctor_slots` | Real-time query |
| Calendar | Google Calendar MCP | Post-call async |
| Email | Email MCP | Post-call async |
| Session state | Redis (Render) | Free tier |
| Persistence | Postgres (Render) | Free tier |
| Bot server | FastAPI + Docker → Render | Free tier |
| Dashboard frontend | React → Vercel | Free tier |
| Training/eval data | ai4bharat/IndicVoices | Open source |
| Server language | Python (asyncio / FastAPI) | — |

---

## 18. Open questions & future work

| Item | Notes |
|---|---|
| **Render spin-down** | UptimeRobot ping mitigates; cold start is ~30s. Document clearly. |
| **Groq rate limits** | Retry queue implemented (§7); log rate limit events in LangSmith for visibility. |
| **Bengali/Marathi confusion** | Evaluate VoxLingua107 confusion matrix on IndicVoices before deploy; document bn/mr pair accuracy specifically. |
| **PII handling** | Phone numbers hashed; consider whether complaint text needs masking for a DPDP-compliant demo. |
| **Render Postgres 97-day expiry** | Export data before expiry; automate with a pg_dump cron if needed. |
| **Context window on long calls** | Add a summarisation step for conversation history beyond N turns to stay within Groq context limits. |
| **Live Calendar availability** | Currently `doctor_slots` is seeded manually. A follow-up could sync with Google Calendar reads to keep it current. |
| **Load testing** | 5×ONNX STT + SpeechBrain + Redis + Postgres under concurrent WebSocket connections — benchmark on Render free tier before claiming production readiness. |
| **Whisper fallback** | If Moonshine Tiny WER is unacceptable for a specific language, swap that model to Faster-Whisper without changing the rest of the pipeline. |