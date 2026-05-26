# ----- Pre-deployment eval for language ID @ backend/scripts/eval_lang_id.py -----

import argparse
import time
from collections.abc import Iterable

import numpy as np
import scipy.signal

from backend.core.language_id import VOXLINGUA_TO_CODE, identify_language

# Map our supported language codes to IndicVoices config names.
INDICVOICES_CONFIGS = {
    "hi": "hindi",
    "te": "telugu",
    "bn": "bengali",
    "mr": "marathi",
}

ENGLISH_DATASET_NAME = "ai4bharat/Svarah"
INDICVOICES_DATASET_NAME = "ai4bharat/IndicVoices"
SAMPLES_PER_LANG = 25
SHUFFLE_BUFFER_SIZE = 256
TARGET_SR = 16000
RANDOM_SEED = 42


def _mu_law_compress(x: np.ndarray) -> np.ndarray:
    mu = 255.0
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def _mu_law_expand(y: np.ndarray) -> np.ndarray:
    mu = 255.0
    return np.sign(y) * (1.0 / mu) * (np.exp(np.abs(y) * np.log1p(mu)) - 1.0)


def telephony_degrade(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply a telephony-channel degradation pipeline to *audio*.

    Steps: 8 kHz resample → 300–3400 Hz bandpass → μ-law compand →
    Gaussian noise (σ=0.002) → 2 % burst-packet-loss → 16 kHz resample.
    """
    if sr != 8000:
        audio = scipy.signal.resample_poly(audio, 8000, sr)
        sr = 8000

    sos = scipy.signal.butter(4, [300, 3400], btype="band", fs=8000, output="sos")
    audio = scipy.signal.sosfilt(sos, audio)

    audio = _mu_law_compress(audio)
    audio = _mu_law_expand(audio)

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.002, audio.shape)
    audio = audio + noise

    frame_len = int(0.020 * 8000)
    for start in range(0, len(audio) - frame_len + 1, frame_len):
        if rng.random() < 0.02:
            audio[start : start + frame_len] = 0.0

    audio = scipy.signal.resample_poly(audio, TARGET_SR, 8000)
    return audio


def _iter_dataset_rows(dataset: Iterable) -> Iterable[dict]:
    """Yield rows from a dataset or dataset-like iterable."""
    for item in dataset:
        yield item


def _load_indicvoices_split(config_name: str):
    """Load one IndicVoices validation split lazily in streaming mode."""
    from datasets import load_dataset

    return load_dataset(
        INDICVOICES_DATASET_NAME,
        name=config_name,
        split="valid",
        streaming=True,
    )


def _load_svarah_split():
    """Load the Svarah English evaluation split lazily in streaming mode."""
    from datasets import load_dataset

    return load_dataset(
        ENGLISH_DATASET_NAME,
        split="test",
        streaming=True,
    )


def _shuffle_dataset(dataset: Iterable):
    """Shuffle a streaming dataset with a bounded in-memory buffer."""
    shuffle = getattr(dataset, "shuffle", None)
    if callable(shuffle):
        return shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=RANDOM_SEED)
    return dataset


def _extract_audio_array(item: dict) -> np.ndarray | None:
    """Extract and resample a single audio row."""
    audio = item.get("audio") or item.get("audio_filepath")
    if not audio:
        return None

    audio_array = audio["array"]
    audio_sr = audio["sampling_rate"]
    if audio_sr != TARGET_SR:
        audio_array = scipy.signal.resample_poly(audio_array, TARGET_SR, audio_sr)

    return audio_array


def _collect_language_samples(dataset: Iterable, sample_cap: int) -> list[np.ndarray]:
    """Collect up to *sample_cap* audio rows from a dataset-like iterable."""
    samples: list[np.ndarray] = []
    for item in _iter_dataset_rows(_shuffle_dataset(dataset)):
        if len(samples) >= sample_cap:
            break

        audio_array = _extract_audio_array(item)
        if audio_array is None:
            continue

        samples.append(audio_array)

    return samples


def load_samples(sample_cap: int = SAMPLES_PER_LANG) -> dict[str, list[np.ndarray]]:
    """Stream balanced evaluation samples for each supported language."""
    samples: dict[str, list[np.ndarray]] = {
        code: [] for code in VOXLINGUA_TO_CODE.values()
    }

    for lang_code, config_name in INDICVOICES_CONFIGS.items():
        samples[lang_code] = _collect_language_samples(
            _load_indicvoices_split(config_name),
            sample_cap=sample_cap,
        )

    samples["en"] = _collect_language_samples(
        _load_svarah_split(),
        sample_cap=sample_cap,
    )

    return samples


def evaluate(
    samples: dict[str, list[np.ndarray]],
) -> tuple[dict[str, dict], float, float]:
    """Run clean and degraded evaluation; return per-language results +
    p50 / p95 latency (ms)."""
    results: dict[str, dict] = {}
    latencies: list[float] = []

    for lang_code, utterances in samples.items():
        correct_clean = 0
        correct_degraded = 0
        total = len(utterances)

        if total == 0:
            results[lang_code] = {
                "clean_acc": 0.0,
                "degraded_acc": 0.0,
                "count": 0,
            }
            continue

        for audio in utterances:
            t0 = time.perf_counter()
            pred_clean, _ = identify_language(audio, sr=TARGET_SR)
            latencies.append((time.perf_counter() - t0) * 1000)

            if pred_clean == lang_code:
                correct_clean += 1

            degraded = telephony_degrade(audio.copy(), TARGET_SR)
            pred_degraded, _ = identify_language(degraded, sr=TARGET_SR)
            if pred_degraded == lang_code:
                correct_degraded += 1

        results[lang_code] = {
            "clean_acc": correct_clean / total * 100,
            "degraded_acc": correct_degraded / total * 100,
            "count": total,
        }

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    return results, p50, p95


def print_table(results: dict, p50: float, p95: float):
    print(
        f"{'Language':<10} {'Clean Acc':<12} {'Degraded Acc':<15} "
        f"{'p50 Latency':<12} {'p95 Latency':<12}"
    )
    print("-" * 61)
    for lang_code in sorted(results):
        r = results[lang_code]
        print(
            f"{lang_code:<10} {r['clean_acc']:<11.1f}% "
            f"{r['degraded_acc']:<14.1f}% {p50:<11.1f}ms {p95:<11.1f}ms"
        )


def main():
    args = parse_args()
    print("Loading streamed evaluation datasets...")
    samples = load_samples(sample_cap=args.samples_per_lang)
    total = sum(len(v) for v in samples.values())
    missing_langs = [
        lang_code
        for lang_code, utterances in samples.items()
        if len(utterances) < args.samples_per_lang
    ]
    print(
        f"Loaded {total} utterances across {len(samples)} target languages "
        f"(cap: {args.samples_per_lang} each)"
    )
    if missing_langs:
        print(
            "Dataset gap: missing full validation coverage for "
            f"{', '.join(sorted(missing_langs))}"
        )

    print("Running evaluation...")
    results, p50, p95 = evaluate(samples)
    print_table(results, p50, p95)

    sample_fails = [
        lc for lc, r in results.items() if r["count"] < args.samples_per_lang
    ]
    clean_fails = [lc for lc, r in results.items() if r["clean_acc"] < 90.0]
    degraded_fails = [lc for lc, r in results.items() if r["degraded_acc"] < 80.0]
    latency_fail = p95 > 100.0

    any_fail = bool(sample_fails or clean_fails or degraded_fails or latency_fail)

    if sample_fails:
        print(
            "\nFAIL: Missing required sample count for: "
            f"{', '.join(sorted(sample_fails))}"
        )

    if clean_fails:
        print(f"\nFAIL: Clean accuracy <90% for: {', '.join(clean_fails)}")
    if degraded_fails:
        print(f"FAIL: Degraded accuracy <80% for: {', '.join(degraded_fails)}")
    if latency_fail:
        print(f"FAIL: p95 latency {p95:.1f}ms exceeds 100ms threshold")

    if any_fail:
        exit(1)
    else:
        print("\nPASS: All acceptance criteria met")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the language-ID evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=SAMPLES_PER_LANG,
        help="Maximum number of utterances to stream per language.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
