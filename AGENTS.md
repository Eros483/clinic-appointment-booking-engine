# AGENT.md

## Project Overview
Real-time multilingual voice agent that answers a clinic phone line via Twilio, identifies the caller's language (hi/en/te/bn/mr), conducts natural conversation to collect appointment details, validates slot availability in Postgres, and schedules the event on Google Calendar — all without human intervention. Resume/portfolio project targeting Tier-1 Indian cities.

**Status:** Backend scaffolding done — full directory tree, `utils/config.py` (Pydantic BaseSettings), `utils/logger.py`, `backend/main.py` (FastAPI + /health), `requirements.txt`, `docs/features.json`. All 33 backend files created as stubs ready for implementation. Frontend directory created but empty.

## Tech Stack (per design doc)
- **Server:** Python 3.13, asyncio / FastAPI, Docker → Render (free tier)
- **Telephony:** Twilio Media Streams (8kHz μ-law G.711)
- **VAD:** Silero VAD (ONNX)
- **Language ID:** SpeechBrain VoxLingua107 ECAPA-TDNN
- **STT:** Moonshine Tiny ONNX Q8 ×5 (hi/en/te/bn/mr)
- **LLM:** Groq API (Llama 3.1 8B)
- **TTS:** Sarvam Bulbul v3
- **Session:** Redis (Render free tier)
- **Persistence:** Postgres + asyncpg (Render free tier)
- **Observability:** LangSmith traces on all LLM calls
- **Dashboard:** React + Vite → Vercel, backed by FastAPI REST
- **Training/eval data:** ai4bharat/IndicVoices

## Key Commands
```bash
uv run backend/main.py            # run FastAPI dev server
uv add <package>                  # add dependency
uv sync                           # sync lockfile
pytest                            # run tests (once implemented)
```

**Formatter:** black (always, once backend dir exists)

## Directory Structure
```
clinic-appointment-booking-engine/
├── backend/
│   ├── __init__.py                 # package marker
│   ├── main.py                     # FastAPI app entry point (uvicorn)
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py           # /api/v1/ prefix router
│   │       ├── calls.py            # GET /calls, GET /calls/{call_sid}
│   │       ├── appointments.py     # GET /appointments
│   │       ├── slots.py            # GET /slots/{doctor_name}
│   │       ├── metrics.py          # GET /metrics/latency, /metrics/languages
│   │       └── twilio.py           # Twilio webhook + Media Stream WS
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio.py                # μ-law decode, resample_poly utilities
│   │   ├── vad.py                  # Silero VAD (ONNX) per-frame + barge-in
│   │   ├── language_id.py          # SpeechBrain VoxLingua107 classifier
│   │   ├── stt.py                  # Moonshine Tiny ONNX Q8 router (5 langs)
│   │   ├── llm.py                  # Groq LLM orchestration + prompt builder
│   │   ├── tts.py                  # Sarvam Bulbul v3 TTS wrapper
│   │   └── state_machine.py        # Call state machine (GREETING→…→DONE)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── calls.py                # calls table queries (asyncpg)
│   │   ├── appointments.py         # appointments table queries
│   │   ├── doctor_slots.py         # doctor_slots table + optimistic lock
│   │   ├── turn_metrics.py         # turn_metrics table writes
│   │   └── language_events.py      # language_events table writes
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── session.py              # Redis session JSON schema / Pydantic model
│   │   └── appointment.py          # Appointment extraction validation
│   ├── services/
│   │   ├── __init__.py
│   │   ├── session.py              # Redis session CRUD (TTL 3600s)
│   │   ├── groq.py                 # Groq client wrapper + LangSmith tracing
│   │   └── calendar.py             # Google Calendar MCP (post-call async)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py               # Pydantic BaseSettings (single source of env)
│   │   └── logger.py               # Structured logging (never print/stdlib)
│   ├── db/
│   │   ├── __init__.py
│   │   ├── pool.py                 # asyncpg connection pool singleton
│   │   └── migrations/
│   │       ├── 001_create_tables.sql
│   │       └── 002_seed_slots.sql
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py             # pytest fixtures (test DB, test Redis, test client)
│   │   ├── test_vad.py
│   │   ├── test_language_id.py
│   │   ├── test_stt.py
│   │   ├── test_llm.py
│   │   ├── test_state_machine.py
│   │   └── test_slot_booking.py
│   └── .env.example
├── frontend/                       # React + Vite dashboard (separate Vercel deploy)
│   ├── src/
│   │   ├── components/             # Reusable UI components (PascalCase.tsx)
│   │   ├── pages/                  # View pages
│   │   ├── services/               # All API calls (never in components)
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   ├── design.md                   # authoritative system spec v2
│   └── features.json               # canonical feature tracker
├── main.py                         # stub entry point (to be removed)
├── pyproject.toml                   # Python >=3.13, uv-managed
├── requirements.txt                 # pip deps for Docker build (uv for dev)
├── .python-version                 # 3.13
└── README.md                       # badges + free-tier constraints + eval results
```

## Conventions

### Python
- **Package manager: `uv`** — use `uv add`, `uv run`, `uv sync`. Never `pip` directly.
  - Exception: the Dockerfile in design.md uses `pip install` for build-time deps — that's fine in CI.
- Every backend file starts with: `# ----- <4-5 word purpose> @ <file location> -----`
- Naming: snake_case for files, variables, functions, DB columns
- All main functions that handle individual components must have precise docstrings written for them. The remaining functions can have a one-liner.
- Formatter: black
- API routes are thin: validate input → call core → return output
- `core/` has zero knowledge of HTTP or FastAPI
- Env vars: use `from utils.config import config` only, never `os.environ`
- Logging: use `from utils.logger import logger` only, never `print` or stdlib `logging`
- Config: Pydantic `BaseSettings` class instantiated once in `backend/utils/config.py`
- **Not SQLite/SQLAlchemy** — design doc specifies Postgres + raw `asyncpg`

### JavaScript (Frontend)
- camelCase for variables/functions, PascalCase for components/types, snake_case for files
- All API calls go through `services/`, never directly in components

### General
- Commits: conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`)
- Never commit secrets; maintain `.env.example` with keys but no values
- API versioned from day one under `/api/v1/`
- README badges: HTML `<img>` tags from shields.io (not Markdown syntax)

## Requirements from design.md

### Mandatory pre-deployment steps
- Validate SpeechBrain language classifier on IndicVoices (clean + telephony-degraded)
- Target metrics: TTFT <400ms, turn latency <800ms, lang accuracy ≥85%
- Must implement: barge-in on VAD, Groq rate-limit retry with "one moment please" TTS, Redis session TTL 3600s, optimistic slot locking (`WHERE status = 'available'`)

### Free-tier constraints to document in README
- Render spins down after 15min inactivity (cold start ~30s) — use UptimeRobot ping on `/health`
- Render Postgres expires at 97 days — export before expiry
- Groq: 30 req/min, 14,400 req/day
- Sarvam TTS: check current limits; cache common phrases at startup
- LangSmith: 5,000 traces/month
- Twilio trial: inbound calls free; announces promo message

### Env vars needed (from design.md)
```
TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, GROQ_API_KEY, SARVAM_API_KEY,
LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2=true, LANGCHAIN_PROJECT=clinic-voice-agent,
DATABASE_URL, REDIS_URL, GOOGLE_CALENDAR_CREDENTIALS_JSON
```

## Agent Roles
Three-agent workflow per task: **Planner** (checks docs/design.md, produces plan, no code) → **Builder** (tests first, then impl, no scope creep) → **Reviewer** (black formatting, snake_case, test coverage, features.json updated). Planner finishes before Builder starts; Reviewer approves before task is done.

## Guidelines
- Always check `docs/design.md` before starting any task — it takes precedence
- Never modify files in `docs/` unless explicitly asked
- Always update `docs/features.json` after completing a task
- Prefer `resample_poly` over `librosa.resample` for real-time Twilio audio chunks
- Flag out-of-scope work rather than silently doing it
- If a design doc is missing for a significant feature, flag it before proceeding
