# ----- Evaluate trained lang ID model @ backend/training/eval_lang_id.py -----

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime

from backend.training.augment import TARGET_SR, simulate_telephony
from backend.training.build_dataset import (
    TARGET_LANGS,
    CACHE_ROOT,
    _load_cached,
)
from backend.training.train_ecapa import LogMelExtractor, N_MELS

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def load_model(
    model_path: str | None = None,
) -> onnxruntime.InferenceSession:
    path = Path(model_path) if model_path else ARTIFACTS_DIR / "lang_id_ecapa_ta.onnx"
    session = onnxruntime.InferenceSession(str(path))
    dummy = np.zeros((1, N_MELS, 100), dtype=np.float32)
    session.run(None, {"mel": dummy})
    return session


def predict(
    session: onnxruntime.InferenceSession, audio: np.ndarray
) -> tuple[str, float]:
    mel = LogMelExtractor.compute(audio, TARGET_SR)
    logits = session.run(None, {"mel": mel})[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    idx = int(np.argmax(probs))
    confidence = float(probs[0][idx])
    return TARGET_LANGS[idx], confidence


def evaluate(
    session: onnxruntime.InferenceSession,
    samples: dict[str, list[np.ndarray]],
    degrade: bool = False,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for lang_code, audios in samples.items():
        correct = 0
        total = len(audios)
        if total == 0:
            results[lang_code] = 0.0
            continue
        for audio in audios:
            if degrade:
                audio = simulate_telephony(audio, TARGET_SR)
            pred, _ = predict(session, audio)
            if pred == lang_code:
                correct += 1
        results[lang_code] = correct / total * 100
    return results


def latency_benchmark(
    session: onnxruntime.InferenceSession,
    samples: dict[str, list[np.ndarray]],
    n: int = 50,
) -> tuple[float, float]:
    all_audios = []
    for audios in samples.values():
        all_audios.extend(audios)
    all_audios = all_audios[:n]

    latencies = []
    for audio in all_audios:
        t0 = time.perf_counter()
        predict(session, audio)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    return p50, p95


def print_results(
    clean: dict[str, float],
    degraded: dict[str, float],
    p50: float,
    p95: float,
):
    print(
        f"{'Language':<10} {'Clean Acc':<12} {'Degraded Acc':<15} "
        f"{'p50':<10} {'p95':<10}"
    )
    print("-" * 57)
    for lang in TARGET_LANGS:
        print(
            f"{lang:<10} {clean.get(lang, 0):<11.1f}% "
            f"{degraded.get(lang, 0):<14.1f}% "
            f"{p50:<9.1f}ms {p95:<9.1f}ms"
        )
    clean_avg = sum(clean.values()) / max(len(clean), 1)
    degraded_avg = sum(degraded.values()) / max(len(degraded), 1)
    print("-" * 57)
    print(f"{'Average':<10} {clean_avg:<11.1f}% {degraded_avg:<14.1f}%")


def _load_test_samples(cache_key: str = "samples_800") -> dict[str, list[np.ndarray]]:
    cache_path = CACHE_ROOT / cache_key
    if cache_path.exists():
        data = _load_cached(cache_path)
        return data.get("test", {})
    print(f"Cache not found at {cache_path}. Streaming fresh data...")
    from backend.training.build_dataset import build_dataset

    data = build_dataset(samples_per_lang=800)
    return data.get("test", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--latency-samples", type=int, default=50)
    args = parser.parse_args()

    print("Loading model...")
    session = load_model(args.model_path)
    print("Loading test samples...")
    test_samples = _load_test_samples()
    total = sum(len(v) for v in test_samples.values())
    print(
        f"Loaded {total} test utterances ({total // max(len(TARGET_LANGS), 1)} per lang)"
    )

    print("Evaluating clean accuracy...")
    clean = evaluate(session, test_samples, degrade=False)

    print("Evaluating telephony-degraded accuracy...")
    degraded = evaluate(session, test_samples, degrade=True)

    print("Running latency benchmark...")
    p50, p95 = latency_benchmark(session, test_samples, n=args.latency_samples)

    print("\n--- Results ---")
    print_results(clean, degraded, p50, p95)

    clean_fails = [lang for lang, acc in clean.items() if acc < 90.0]
    degraded_fails = [lang for lang, acc in degraded.items() if acc < 80.0]
    latency_fail = p95 > 100.0

    any_fail = bool(clean_fails or degraded_fails or latency_fail)
    if clean_fails:
        print(f"\nFAIL: Clean acc <90% for: {', '.join(clean_fails)}")
    if degraded_fails:
        print(f"FAIL: Degraded acc <80% for: {', '.join(degraded_fails)}")
    if latency_fail:
        print(f"FAIL: p95 latency {p95:.1f}ms exceeds 100ms threshold")
    if any_fail:
        raise SystemExit(1)
    print("\nPASS: All acceptance criteria met")


if __name__ == "__main__":
    main()
