from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "0_data_pre_processing"
DEFAULT_MODEL_INPUT_PATH = (
    DATA_ROOT / "processed_dataset" / "model_input" / "model_input_dataset.parquet"
)
DEFAULT_MEL_TENSOR_DIR = DATA_ROOT / "processed_dataset" / "mel_tensors"
DEFAULT_CHUNK_COUNT = 50
DEFAULT_CHUNK_SIZE = 64

MODEL_INPUT_COLUMNS = [
    "music_id",
    "music_name",
    "lyrics_e5_large_embedding",
    "main_genre_embedding",
    "music_lang_embedding",
    "genre_condition_embedding",
    "language_condition_embedding",
    "condition_embedding",
    "art_id",
    "art_name",
]


def _as_float_tensor(value: Any) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


class MusicVAEDatasetTill50(Dataset):
    """Training dataset backed by the first 50 pre-extracted mel tensor chunks."""

    def __init__(
        self,
        model_input_path: str | Path = DEFAULT_MODEL_INPUT_PATH,
        mel_tensor_dir: str | Path = DEFAULT_MEL_TENSOR_DIR,
        chunk_count: int = DEFAULT_CHUNK_COUNT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        mel_cache_chunks: int = 2,
    ) -> None:
        self.model_input_path = Path(model_input_path)
        self.mel_tensor_dir = Path(mel_tensor_dir)
        self.chunk_count = chunk_count
        self.chunk_size = chunk_size
        self.mel_cache_chunks = mel_cache_chunks

        self.chunk_paths = [
            self.mel_tensor_dir / f"chunk_{index:05d}.pt" for index in range(chunk_count)
        ]
        missing = [path for path in self.chunk_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing mel tensor chunk: {missing[0]}")

        self.length = chunk_count * chunk_size
        self.features = pq.read_table(
            self.model_input_path,
            columns=MODEL_INPUT_COLUMNS,
        ).slice(0, self.length)
        if self.features.num_rows < self.length:
            self.length = self.features.num_rows
        self._columns = {name: self.features[name] for name in MODEL_INPUT_COLUMNS}
        self._mel_cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return self.length

    def _feature_value(self, column: str, index: int) -> Any:
        return self._columns[column][index].as_py()

    def _load_mel_chunk(self, chunk_index: int) -> torch.Tensor:
        return torch.load(
            self.chunk_paths[chunk_index],
            map_location="cpu",
            weights_only=True,
        )

    def _mel_tensor(self, index: int) -> torch.Tensor:
        chunk_index = index // self.chunk_size
        offset = index % self.chunk_size

        if chunk_index not in self._mel_cache:
            self._mel_cache[chunk_index] = self._load_mel_chunk(chunk_index)
            self._mel_cache.move_to_end(chunk_index)
            while len(self._mel_cache) > self.mel_cache_chunks:
                self._mel_cache.popitem(last=False)
        else:
            self._mel_cache.move_to_end(chunk_index)

        return self._mel_cache[chunk_index][offset].float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "music_id": self._feature_value("music_id", index),
            "music_name": self._feature_value("music_name", index),
            "art_id": self._feature_value("art_id", index),
            "art_name": self._feature_value("art_name", index),
            "lyrics_e5_large_embedding": _as_float_tensor(
                self._feature_value("lyrics_e5_large_embedding", index)
            ),
            "melspectrogram": self._mel_tensor(index),
            "main_genre_embedding": _as_float_tensor(
                self._feature_value("main_genre_embedding", index)
            ),
            "music_lang_embedding": _as_float_tensor(
                self._feature_value("music_lang_embedding", index)
            ),
            "genre_condition_embedding": _as_float_tensor(
                self._feature_value("genre_condition_embedding", index)
            ),
            "language_condition_embedding": _as_float_tensor(
                self._feature_value("language_condition_embedding", index)
            ),
            "condition_embedding": _as_float_tensor(
                self._feature_value("condition_embedding", index)
            ),
        }


def music_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = [
        "lyrics_e5_large_embedding",
        "melspectrogram",
        "main_genre_embedding",
        "music_lang_embedding",
        "genre_condition_embedding",
        "language_condition_embedding",
        "condition_embedding",
    ]
    text_keys = ["music_id", "music_name", "art_id", "art_name"]

    output: dict[str, Any] = {}
    for key in tensor_keys:
        output[key] = torch.stack([item[key] for item in batch], dim=0)
    for key in text_keys:
        output[key] = [item[key] for item in batch]
    return output


def create_dataloader_till_50(
    model_input_path: str | Path = DEFAULT_MODEL_INPUT_PATH,
    mel_tensor_dir: str | Path = DEFAULT_MEL_TENSOR_DIR,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    mel_cache_chunks: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = MusicVAEDatasetTill50(
        model_input_path=model_input_path,
        mel_tensor_dir=mel_tensor_dir,
        mel_cache_chunks=mel_cache_chunks,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=music_collate_fn,
    )


if __name__ == "__main__":
    dataset = MusicVAEDatasetTill50()
    print(f"dataset_len: {len(dataset)}")
    loader = create_dataloader_till_50(batch_size=4)
    batch = next(iter(loader))
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key}: {value[:4]}")
