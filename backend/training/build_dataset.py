# ----- Streaming dataset builder for lang ID @ backend/training/build_dataset.py -----

import json
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal

CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache" / "training_dataset"
TARGET_SR = 16000
RANDOM_SEED = 42
SHUFFLE_BUFFER = 256

INDICVOICES_NAME = "ai4bharat/IndicVoices"
INDICVOICES_CONFIGS = {
    "hi": "hindi",
    "ta": "tamil",
    "bn": "bengali",
    "mr": "marathi",
}
SVARAH_NAME = "ai4bharat/Svarah"

TARGET_LANGS = ["hi", "en", "ta", "bn", "mr"]


def _resample(audio_array: np.ndarray, sr_in: int) -> np.ndarray:
    arr = np.asarray(audio_array, dtype=np.float32)
    if sr_in != TARGET_SR:
        arr = scipy.signal.resample_poly(arr, TARGET_SR, sr_in)
    return arr


def _extract_audio(row: dict) -> Optional[np.ndarray]:
    audio = row.get("audio") or row.get("audio_filepath")
    if audio is None:
        return None
    arr = _resample(audio["array"], audio["sampling_rate"])
    if len(arr) < TARGET_SR // 2:
        return None
    return arr


def build_dataset(
    samples_per_lang: int = 800,
    val_split: float = 0.1,
    test_split: float = 0.1,
    force_rebuild: bool = False,
) -> dict[str, dict[str, list[np.ndarray]]]:
    """Build train/val/test splits from IndicVoices + Svarah.

    Returns nested dict: {split: {lang_code: [audio_arrays]}}
    where split is one of "train", "val", "test".
    """
    from datasets import load_dataset

    cache_path = CACHE_ROOT / f"samples_{samples_per_lang}"
    metadata_path = cache_path / "_metadata.json"

    if cache_path.exists() and not force_rebuild:
        return _load_cached(cache_path)

    all_samples: dict[str, list[np.ndarray]] = {lang: [] for lang in TARGET_LANGS}

    for lang_code, config_name in INDICVOICES_CONFIGS.items():
        stream = load_dataset(
            INDICVOICES_NAME, name=config_name, split="train", streaming=True
        )
        stream = stream.shuffle(buffer_size=SHUFFLE_BUFFER, seed=RANDOM_SEED)
        count = 0
        for row in stream:
            if count >= samples_per_lang:
                break
            audio = _extract_audio(row)
            if audio is not None:
                all_samples[lang_code].append(audio)
                count += 1

    stream = load_dataset(SVARAH_NAME, split="test", streaming=True)
    stream = stream.shuffle(buffer_size=SHUFFLE_BUFFER, seed=RANDOM_SEED)
    count = 0
    for row in stream:
        if count >= samples_per_lang:
            break
        audio = _extract_audio(row)
        if audio is not None:
            all_samples["en"].append(audio)
            count += 1

    if len(all_samples["en"]) == 0:
        print(
            "WARNING: No English samples found from Svarah (split='test'). "
            "Trying split='train'..."
        )
        stream = load_dataset(SVARAH_NAME, split="train", streaming=True)
        stream = stream.shuffle(buffer_size=SHUFFLE_BUFFER, seed=RANDOM_SEED)
        count = 0
        for row in stream:
            if count >= samples_per_lang:
                break
            audio = _extract_audio(row)
            if audio is not None:
                all_samples["en"].append(audio)
                count += 1

    splits: dict[str, dict[str, list[np.ndarray]]] = {
        "train": {lang: [] for lang in TARGET_LANGS},
        "val": {lang: [] for lang in TARGET_LANGS},
        "test": {lang: [] for lang in TARGET_LANGS},
    }

    rng = np.random.default_rng(RANDOM_SEED)
    for lang_code in TARGET_LANGS:
        audios = all_samples[lang_code]
        rng.shuffle(audios)
        n = len(audios)
        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))
        n_train = n - n_test - n_val

        splits["test"][lang_code] = audios[:n_test]
        splits["val"][lang_code] = audios[n_test : n_test + n_val]
        splits["train"][lang_code] = audios[n_test + n_val :]

    _save_cached(cache_path, splits)
    return splits


def _save_cached(path: Path, splits: dict[str, dict[str, list[np.ndarray]]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for split_name, lang_dict in splits.items():
        for lang_code, audios in lang_dict.items():
            key = f"{split_name}_{lang_code}"
            dir_path = path / key
            dir_path.mkdir(parents=True, exist_ok=True)
            for i, arr in enumerate(audios):
                np.save(dir_path / f"{i}.npy", arr)
            metadata[key] = len(audios)
    with open(path / "_metadata.json", "w") as f:
        json.dump(metadata, f)


def _load_cached(path: Path) -> dict[str, dict[str, list[np.ndarray]]]:
    with open(path / "_metadata.json") as f:
        metadata = json.load(f)
    splits: dict[str, dict[str, list[np.ndarray]]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    for key, count in metadata.items():
        split_name, lang_code = key.split("_", 1)
        dir_path = path / key
        audios = []
        for i in range(count):
            audios.append(np.load(dir_path / f"{i}.npy"))
        if split_name not in splits:
            splits[split_name] = {}
        splits[split_name][lang_code] = audios
    return splits


def print_split_summary(splits: dict[str, dict[str, list[np.ndarray]]]) -> None:
    print(f"{'Split':<8} {' | '.join(f'{lang:<6}' for lang in TARGET_LANGS)}")
    print("-" * 50)
    for split_name in ["train", "val", "test"]:
        counts = [str(len(splits[split_name].get(lang, []))) for lang in TARGET_LANGS]
        print(f"{split_name:<8} {' | '.join(f'{c:<6}' for c in counts)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-lang", type=int, default=800)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    splits = build_dataset(
        samples_per_lang=args.samples_per_lang,
        force_rebuild=args.force_rebuild,
    )
    print_split_summary(splits)
