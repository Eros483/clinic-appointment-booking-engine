# ----- Eval dataset caching utils @ backend/utils/eval_dataset_cache.py -----

from pathlib import Path

import numpy as np
import scipy.signal

from backend.utils.logger import logger

INDICVOICES_CONFIGS = {
    "hi": "hindi",
    "te": "telugu",
    "bn": "bengali",
    "mr": "marathi",
}
ENGLISH_DATASET_NAME = "ai4bharat/Svarah"
INDICVOICES_DATASET_NAME = "ai4bharat/IndicVoices"
SHUFFLE_BUFFER_SIZE = 256
TARGET_SR = 16000
RANDOM_SEED = 42
DEFAULT_CACHE_ROOT = Path("backend/.cache/lang_id_eval")


def _load_dataset_module():
    """Import datasets lazily so unit tests do not require it."""
    from datasets import Dataset, load_dataset, load_from_disk

    return Dataset, load_dataset, load_from_disk


def _cache_dir(sample_cap: int, cache_root: Path | None = None) -> Path:
    """Build the on-disk cache path for a given sample cap."""
    root = cache_root or DEFAULT_CACHE_ROOT
    return root / f"samples_{sample_cap}"


def _load_indicvoices_split(config_name: str):
    """Load one IndicVoices split in streaming mode."""
    _, load_dataset, _ = _load_dataset_module()
    return load_dataset(
        INDICVOICES_DATASET_NAME,
        name=config_name,
        split="valid",
        streaming=True,
    )


def _load_svarah_split():
    """Load the Svarah English split in streaming mode."""
    _, load_dataset, _ = _load_dataset_module()
    return load_dataset(
        ENGLISH_DATASET_NAME,
        split="test",
        streaming=True,
    )


def _shuffle_and_take(dataset, sample_cap: int):
    """Shuffle a streaming dataset and take a bounded sample."""
    return dataset.shuffle(
        buffer_size=SHUFFLE_BUFFER_SIZE,
        seed=RANDOM_SEED,
    ).take(sample_cap)


def _resample_audio(audio_array, sampling_rate: int) -> np.ndarray:
    """Resample audio to 16 kHz for language-ID evaluation."""
    array = np.asarray(audio_array, dtype=np.float32)
    if sampling_rate != TARGET_SR:
        array = scipy.signal.resample_poly(array, TARGET_SR, sampling_rate)
    return array


def _serialize_row(item: dict, language_code: str) -> dict | None:
    """Convert one raw dataset row into a cacheable evaluation example."""
    audio = item.get("audio") or item.get("audio_filepath")
    if not audio:
        logger.warning(f"Skipping item without audio for {language_code}")
        return None

    audio_array = _resample_audio(
        audio["array"],
        audio["sampling_rate"],
    )
    return {
        "language_code": language_code,
        "audio_array": audio_array.tolist(),
        "sampling_rate": TARGET_SR,
    }


def build_eval_dataset_cache(
    sample_cap: int,
    cache_root: Path | None = None,
):
    """Stream, sample, and save a local evaluation dataset cache."""
    Dataset, _, _ = _load_dataset_module()

    rows: list[dict] = []
    for language_code, config_name in INDICVOICES_CONFIGS.items():
        stream = _load_indicvoices_split(config_name)
        for item in _shuffle_and_take(stream, sample_cap):
            row = _serialize_row(item, language_code)
            if row is not None:
                rows.append(row)

    english_stream = _load_svarah_split()
    for item in _shuffle_and_take(english_stream, sample_cap):
        row = _serialize_row(item, "en")
        if row is not None:
            rows.append(row)

    dataset = Dataset.from_list(rows)
    cache_dir = _cache_dir(sample_cap, cache_root=cache_root)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_dir))
    logger.info(f"Saved eval dataset cache to {cache_dir}")
    return dataset


def load_or_build_eval_dataset(
    sample_cap: int,
    cache_root: Path | None = None,
    refresh: bool = False,
):
    """Load the local evaluation dataset cache, or build it on demand."""
    _, _, load_from_disk = _load_dataset_module()
    cache_dir = _cache_dir(sample_cap, cache_root=cache_root)

    if cache_dir.exists() and not refresh:
        logger.info(f"Loading cached eval dataset from {cache_dir}")
        return load_from_disk(str(cache_dir))

    return build_eval_dataset_cache(
        sample_cap=sample_cap,
        cache_root=cache_root,
    )


def dataset_to_samples(dataset) -> dict[str, list[np.ndarray]]:
    """Convert a saved Hugging Face dataset into eval-ready numpy arrays."""
    samples = {"hi": [], "en": [], "te": [], "bn": [], "mr": []}
    for item in dataset:
        language_code = item["language_code"]
        if language_code not in samples:
            continue
        samples[language_code].append(
            np.asarray(item["audio_array"], dtype=np.float32)
        )
    return samples
