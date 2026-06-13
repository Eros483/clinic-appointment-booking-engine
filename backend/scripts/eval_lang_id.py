# ----- Pre-deployment eval for language ID @ backend/scripts/eval_lang_id.py -----

import argparse
import time

import numpy as np
import scipy.signal

from backend.core.language_id import identify_language
from backend.utils.eval_dataset_cache import (
    TARGET_SR,
    dataset_to_samples,
    load_or_build_eval_dataset,
)

SAMPLES_PER_LANG = 25


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


def load_samples(
    sample_cap: int = SAMPLES_PER_LANG,
    refresh_cache: bool = False,
) -> dict[str, list[np.ndarray]]:
    """Load eval samples from a small local cached Hugging Face dataset."""
    dataset = load_or_build_eval_dataset(
        sample_cap=sample_cap,
        refresh=refresh_cache,
    )
    return dataset_to_samples(dataset)


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

            pred_clean, _, raw_clean = identify_language(audio, sr=TARGET_SR)
            latencies.append((time.perf_counter() - t0) * 1000)

            if lang_code == "en" and pred_clean != "en":
                print(f"[DEBUG] English audio misclassified. Raw label: {raw_clean}")

            if pred_clean == lang_code:
                correct_clean += 1

            degraded = telephony_degrade(audio.copy(), TARGET_SR)
            pred_degraded, _, _ = identify_language(degraded, sr=TARGET_SR)
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
    print("Loading cached evaluation dataset...")
    samples = load_samples(
        sample_cap=args.samples_per_lang,
        refresh_cache=args.refresh_cache,
    )
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
        raise SystemExit(1)

    print("\nPASS: All acceptance criteria met")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the language-ID evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-per-lang",
        type=int,
        default=SAMPLES_PER_LANG,
        help="Maximum number of utterances to cache per language.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild the cached evaluation dataset from the streaming sources.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
