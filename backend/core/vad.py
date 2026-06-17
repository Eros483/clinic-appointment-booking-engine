# ----- Silero VAD (ONNX) per-frame processor @ backend/core/vad.py -----

import numpy as np
import torch

from backend.utils.logger import logger

VAD_START = "VAD_START"
VAD_END = "VAD_END"
BARGE_IN = "BARGE_IN"

SAMPLE_RATE = 16000
FRAME_SIZE = 512
SILENCE_THRESHOLD_FRAMES = 44
PRE_ROLL_FRAMES = 19
SPEECH_PROB_THRESHOLD = 0.5

_model = None


def _load_model():
    global _model
    try:
        from silero_vad import load_silero_vad

        _model = load_silero_vad(onnx=True)
        logger.info("Silero VAD ONNX model loaded via silero-vad package")
    except Exception:
        logger.warning("silero-vad model unavailable — VAD disabled")
        _model = None


class VADProcessor:
    """Per-call Silero VAD processor with pre-roll buffer and barge-in.

    Maintains utterance buffer and circular pre-roll buffer. Delegates
    per-frame inference to the ``silero-vad`` package's ONNX wrapper.
    Caller should create one instance per Twilio call.
    """

    def __init__(self) -> None:
        self.state = "idle"
        self.silence_counter = 0
        self.pre_roll: list[np.ndarray] = []
        self.utterance: list[np.ndarray] = []

    def _get_speech_prob(self, frame: np.ndarray) -> float:
        global _model
        if _model is None:
            _load_model()
        if _model is None:
            return 0.0

        x = torch.from_numpy(frame).float()
        out = _model(x, SAMPLE_RATE)
        return float(out[0][0])

    def _update_pre_roll(self, frame: np.ndarray) -> None:
        self.pre_roll.append(frame)
        if len(self.pre_roll) > PRE_ROLL_FRAMES:
            self.pre_roll.pop(0)

    def reset(self) -> None:
        self.state = "idle"
        self.silence_counter = 0
        self.pre_roll.clear()
        self.utterance.clear()
        if _model is not None:
            _model.reset_states()

    def get_utterance_audio(self) -> np.ndarray:
        if not self.utterance:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.utterance)

    def process(self, frame: np.ndarray, is_tts_active: bool = False) -> str | None:
        if frame.shape != (FRAME_SIZE,) and frame.shape != (1, FRAME_SIZE):
            raise ValueError(f"frame must be {FRAME_SIZE} samples, got {frame.shape}")

        frame = frame.ravel()
        self._update_pre_roll(frame)
        prob = self._get_speech_prob(frame)

        is_speech = prob > SPEECH_PROB_THRESHOLD

        if is_speech and is_tts_active:
            return self._on_barge_in(frame)

        if is_speech and self.state == "idle":
            return self._on_speech_start(frame)

        if is_speech and self.state == "speaking":
            self.utterance.append(frame)
            return None

        if not is_speech and self.state == "speaking":
            self.utterance.append(frame)
            self.silence_counter += 1
            if self.silence_counter >= SILENCE_THRESHOLD_FRAMES:
                return self._on_speech_end()
            return None

        return None

    def _on_speech_start(self, frame: np.ndarray) -> str:
        self.state = "speaking"
        self.silence_counter = 0
        self.utterance = list(self.pre_roll) + [frame]
        return VAD_START

    def _on_barge_in(self, frame: np.ndarray) -> str:
        if _model is not None:
            _model.reset_states()
        self.state = "speaking"
        self.silence_counter = 0
        self.utterance = list(self.pre_roll) + [frame]
        return BARGE_IN

    def _on_speech_end(self) -> str:
        self.state = "idle"
        self.silence_counter = 0
        return VAD_END


__all__ = [
    "VADProcessor",
    "VAD_START",
    "VAD_END",
    "BARGE_IN",
    "SAMPLE_RATE",
    "FRAME_SIZE",
    "SPEECH_PROB_THRESHOLD",
    "SILENCE_THRESHOLD_FRAMES",
    "PRE_ROLL_FRAMES",
]
