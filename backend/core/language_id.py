# ----- Custom ECAPA-TDNN Lang ID (ONNX) @ backend/core/language_id.py -----

from pathlib import Path

import numpy as np

from backend.utils.logger import logger

LANG_CODES = ["hi", "en", "ta", "bn", "mr"]
_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "lang_id_ecapa_ta.onnx"
)

N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
TARGET_SR = 16000

_session = None


def _load_model() -> None:
    global _session
    try:
        import onnxruntime
    except ImportError:
        logger.warning("onnxruntime not available — lang ID disabled")
        _session = None
        return

    if not _MODEL_PATH.exists():
        logger.warning(f"Lang ID model not found at {_MODEL_PATH}")
        _session = None
        return

    _session = onnxruntime.InferenceSession(str(_MODEL_PATH))
    dummy = np.zeros((1, N_MELS, 100), dtype=np.float32)
    _session.run(None, {"mel": dummy})


def _mel_filterbank(sr: int) -> np.ndarray:
    import math

    low = 0.0
    high = sr / 2
    mel_low = 2595.0 * math.log10(1.0 + low / 700.0)
    mel_high = 2595.0 * math.log10(1.0 + high / 700.0)
    mel_points = np.linspace(mel_low, mel_high, N_MELS + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((N_FFT + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, N_FFT // 2)

    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        left = int(bins[m - 1])
        center = int(bins[m])
        right = int(bins[m + 1])
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


_fb_cache: np.ndarray | None = None


def _ensure_fb(sr: int = TARGET_SR) -> np.ndarray:
    global _fb_cache
    if _fb_cache is None:
        _fb_cache = _mel_filterbank(sr)
    return _fb_cache


def _log_mel(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    fb = _ensure_fb(sr)
    n_fft = N_FFT
    hop = HOP_LENGTH
    win = WIN_LENGTH
    window = np.hanning(win).astype(np.float32)

    n_frames = 1 + (len(audio) - win) // hop
    if n_frames < 1:
        return np.zeros((1, N_MELS, 1), dtype=np.float32)

    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop
        segment = audio[start : start + win] * window
        stft[:, i] = np.fft.rfft(segment, n=n_fft)

    power = np.abs(stft) ** 2
    mel = fb @ power
    log_mel = np.log(np.clip(mel, 1e-10, None))
    return log_mel[np.newaxis, :, :].astype(np.float32)


def identify_language(audio: np.ndarray, sr: int = 16000) -> tuple[str, float, str]:
    """Identify the language of an audio utterance.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio waveform as a 1-D float32 array (values in [-1, 1]).
    sr : int
        Sampling rate of ``audio`` (default 16000).

    Returns
    -------
    lang_code : str
        One of ``"hi"``, ``"en"``, ``"ta"``, ``"bn"``, ``"mr"``.
        Falls back to ``"hi"`` when model is unavailable.
    confidence : float
        Softmax probability of the top prediction (0–1 range).
    raw_label : str
        The top language code (same as lang_code); kept for API compatibility.
    """
    global _session

    if _session is None:
        _load_model()

    if _session is None:
        return "hi", 0.0, "hi"

    if sr != TARGET_SR:
        import scipy.signal

        audio = scipy.signal.resample_poly(audio, TARGET_SR, sr)

    mel = _log_mel(audio, TARGET_SR)
    logits = _session.run(None, {"mel": mel})[0]

    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp / np.sum(exp, axis=-1, keepdims=True)

    idx = int(np.argmax(probs))
    confidence = float(probs[0][idx])
    lang_code = LANG_CODES[idx]
    return lang_code, confidence, lang_code


def update_active_language(
    prediction: str,
    confidence: float,
    word_count: int,
    current_language: str,
    is_first_utterance: bool,
) -> str:
    """Decide whether to switch the active call language.

    Parameters
    ----------
    prediction : str
        Language code predicted by :func:`identify_language`.
    confidence : float
        Confidence of the prediction (0–1).
    word_count : int
        Number of words in the current utterance.
    current_language : str
        Currently active language for this call.
    is_first_utterance : bool
        Whether this is the first utterance in the call.

    Returns
    -------
    str
        The language code to use for subsequent processing.
    """
    if is_first_utterance and confidence < 0.80:
        return "hi"
    if confidence >= 0.80 and word_count > 5:
        return prediction
    return current_language
