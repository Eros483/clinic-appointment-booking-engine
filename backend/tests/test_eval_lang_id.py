# ----- Eval script loader tests @ backend/tests/test_eval_lang_id.py -----

from unittest.mock import patch

import numpy as np

from backend.scripts import eval_lang_id


class FakeStreamingDataset:
    def __init__(self, rows):
        self._rows = rows
        self.shuffle_calls = []

    def shuffle(self, *, buffer_size: int, seed: int):
        self.shuffle_calls.append((buffer_size, seed))
        return self

    def __iter__(self):
        return iter(self._rows)


def test_load_samples_uses_per_language_configs_and_svarah_for_english():
    indic_calls = []
    svarah_dataset = FakeStreamingDataset(
        [
            {
                "audio": {
                    "array": np.zeros(8000, dtype=np.float32),
                    "sampling_rate": 8000,
                }
            }
        ]
    )

    def fake_load_indicvoices_split(name: str):
        indic_calls.append(name)
        return FakeStreamingDataset(
            [
                {
                    "audio": {
                        "array": np.zeros(8000, dtype=np.float32),
                        "sampling_rate": 8000,
                    }
                }
            ]
        )

    with patch(
        "backend.scripts.eval_lang_id._load_indicvoices_split",
        side_effect=fake_load_indicvoices_split,
    ), patch(
        "backend.scripts.eval_lang_id._load_svarah_split",
        return_value=svarah_dataset,
    ):
        samples = eval_lang_id.load_samples(sample_cap=1)

    assert indic_calls == [
        "hindi",
        "telugu",
        "bengali",
        "marathi",
    ]
    assert len(samples["hi"]) == 1
    assert len(samples["te"]) == 1
    assert len(samples["bn"]) == 1
    assert len(samples["mr"]) == 1
    assert len(samples["en"]) == 1
    assert svarah_dataset.shuffle_calls == [
        (eval_lang_id.SHUFFLE_BUFFER_SIZE, eval_lang_id.RANDOM_SEED)
    ]


def test_load_samples_resamples_audio_to_target_rate():
    svarah_dataset = FakeStreamingDataset(
        [
            {
                "audio": {
                    "array": np.zeros(8000, dtype=np.float32),
                    "sampling_rate": 8000,
                }
            }
        ]
    )
    with patch(
        "backend.scripts.eval_lang_id._load_indicvoices_split",
        return_value=FakeStreamingDataset(
            [
                {
                    "audio": {
                        "array": np.zeros(8000, dtype=np.float32),
                        "sampling_rate": 8000,
                    }
                }
            ]
        ),
    ), patch(
        "backend.scripts.eval_lang_id._load_svarah_split",
        return_value=svarah_dataset,
    ):
        samples = eval_lang_id.load_samples(sample_cap=1)

    assert samples["hi"][0].shape[0] == eval_lang_id.TARGET_SR


def test_load_samples_caps_each_language_equally():
    rows = [
        {
            "audio": {
                "array": np.zeros(16000, dtype=np.float32),
                "sampling_rate": 16000,
            }
        }
        for _ in range(3)
    ]
    with patch(
        "backend.scripts.eval_lang_id._load_indicvoices_split",
        return_value=FakeStreamingDataset(rows),
    ), patch(
        "backend.scripts.eval_lang_id._load_svarah_split",
        return_value=FakeStreamingDataset(rows),
    ):
        samples = eval_lang_id.load_samples(sample_cap=2)

    assert all(len(utterances) == 2 for utterances in samples.values())
