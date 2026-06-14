# ----- Eval script cache tests @ backend/tests/test_eval_lang_id.py -----

from unittest.mock import patch

import numpy as np

from backend.scripts import eval_lang_id


def test_load_samples_reads_from_cached_dataset():
    fake_dataset = [
        {"language_code": "hi", "audio_array": [0.0, 1.0]},
        {"language_code": "en", "audio_array": [1.0, 0.0]},
    ]
    with (
        patch(
            "backend.scripts.eval_lang_id.load_or_build_eval_dataset",
            return_value=fake_dataset,
        ),
        patch(
            "backend.scripts.eval_lang_id.dataset_to_samples",
            return_value={
                "hi": [np.zeros(2, dtype=np.float32)],
                "en": [np.ones(2, dtype=np.float32)],
                "ta": [],
                "bn": [],
                "mr": [],
            },
        ) as mock_to_samples,
    ):
        samples = eval_lang_id.load_samples(sample_cap=1, refresh_cache=True)

    assert len(samples["hi"]) == 1
    assert len(samples["en"]) == 1
    mock_to_samples.assert_called_once_with(fake_dataset)


def test_load_samples_forwards_cache_flags():
    with (
        patch(
            "backend.scripts.eval_lang_id.load_or_build_eval_dataset",
            return_value=[],
        ) as mock_loader,
        patch(
            "backend.scripts.eval_lang_id.dataset_to_samples",
            return_value={"hi": [], "en": [], "ta": [], "bn": [], "mr": []},
        ),
    ):
        eval_lang_id.load_samples(sample_cap=7, refresh_cache=True)

    mock_loader.assert_called_once_with(sample_cap=7, refresh=True)
