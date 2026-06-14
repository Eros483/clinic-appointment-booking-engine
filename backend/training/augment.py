# ----- Telephony degradation augmentation @ backend/training/augment.py -----

import numpy as np
import scipy.signal

TARGET_SR = 16000
RNG = np.random.default_rng(42)


def _mu_law_compress(x: np.ndarray) -> np.ndarray:
    mu = 255.0
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def _mu_law_expand(y: np.ndarray) -> np.ndarray:
    mu = 255.0
    return np.sign(y) * (1.0 / mu) * (np.exp(np.abs(y) * np.log1p(mu)) - 1.0)


def simulate_telephony(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Apply telephony-channel degradation to a clean audio waveform.

    Steps: 8 kHz resample → 300–3400 Hz bandpass → μ-law compand →
    Gaussian noise (σ=0.002) → 2 % burst-packet-loss → 16 kHz resample.
    """
    if sr != 8000:
        audio = scipy.signal.resample_poly(audio, 8000, sr)

    sos = scipy.signal.butter(4, [300, 3400], btype="band", fs=8000, output="sos")
    audio = scipy.signal.sosfilt(sos, audio)

    audio = _mu_law_compress(audio)
    audio = _mu_law_expand(audio)

    noise = RNG.normal(0, 0.002, audio.shape)
    audio = audio + noise

    frame_len = int(0.020 * 8000)
    for start in range(0, len(audio) - frame_len + 1, frame_len):
        if RNG.random() < 0.02:
            audio[start : start + frame_len] = 0.0

    audio = scipy.signal.resample_poly(audio, TARGET_SR, 8000)
    return audio


def random_augment(
    audio: np.ndarray, sr: int = TARGET_SR, prob: float = 0.6
) -> np.ndarray:
    """Apply telephony augmentation with given probability."""
    if RNG.random() < prob:
        return simulate_telephony(audio, sr)
    return audio
