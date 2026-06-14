# ----- Eval dataset cache tests @ backend/tests/test_eval_dataset_cache.py -----

from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.utils import eval_dataset_cache


class FakeStreamingDataset:
    def __init__(self, rows):
        self._rows = rows
        self.shuffle_calls = []
        self.take_calls = []

    def shuffle(self, *, buffer_size: int, seed: int):
        self.shuffle_calls.append((buffer_size, seed))
        return self

    def take(self, sample_cap: int):
        self.take_calls.append(sample_cap)
        return self._rows[:sample_cap]


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.saved_path = None

    def save_to_disk(self, path: str):
        self.saved_path = path

    def __iter__(self):
        return iter(self.rows)


def test_build_eval_dataset_cache_streams_shuffle_take_and_save(tmp_path):
    rows = [
        {
            "audio": {
                "array": np.zeros(8000, dtype=np.float32),
                "sampling_rate": 8000,
            }
        }
        for _ in range(3)
    ]
    fake_stream = FakeStreamingDataset(rows)
    fake_dataset = FakeDataset([])

    class FakeDatasetFactory:
        @staticmethod
        def from_list(serialized_rows):
            fake_dataset.rows = serialized_rows
            return fake_dataset

    with (
        patch(
            "backend.utils.eval_dataset_cache._load_dataset_module",
            return_value=(FakeDatasetFactory, None, None),
        ),
        patch(
            "backend.utils.eval_dataset_cache._load_indicvoices_split",
            return_value=fake_stream,
        ),
        patch(
            "backend.utils.eval_dataset_cache._load_svarah_split",
            return_value=fake_stream,
        ),
    ):
        dataset = eval_dataset_cache.build_eval_dataset_cache(
            sample_cap=2,
            cache_root=tmp_path,
        )

    assert dataset is fake_dataset
    assert fake_stream.shuffle_calls
    assert fake_stream.take_calls == [2, 2, 2, 2, 2]
    assert len(fake_dataset.rows) == 10
    assert Path(fake_dataset.saved_path) == tmp_path / "samples_2"


def test_load_or_build_eval_dataset_uses_saved_cache(tmp_path):
    cache_dir = tmp_path / "samples_3"
    cache_dir.mkdir(parents=True)

    with patch(
        "backend.utils.eval_dataset_cache._load_dataset_module",
        return_value=(None, None, lambda path: {"path": path}),
    ):
        dataset = eval_dataset_cache.load_or_build_eval_dataset(
            sample_cap=3,
            cache_root=tmp_path,
        )

    assert dataset == {"path": str(cache_dir)}


def test_dataset_to_samples_groups_rows_by_language():
    dataset = [
        {"language_code": "hi", "audio_array": [0.0, 1.0]},
        {"language_code": "en", "audio_array": [1.0, 0.0]},
        {"language_code": "hi", "audio_array": [2.0, 3.0]},
    ]

    samples = eval_dataset_cache.dataset_to_samples(dataset)

    assert len(samples["hi"]) == 2
    assert len(samples["en"]) == 1
    assert samples["hi"][0].dtype == np.float32
