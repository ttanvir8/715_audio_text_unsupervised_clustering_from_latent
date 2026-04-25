from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "0_data_pre_processing"
DEFAULT_MODEL_INPUT_PATH = (
    DATA_ROOT / "processed_dataset" / "model_input" / "model_input_dataset.parquet"
)
DEFAULT_MEL_TENSOR_DIR = DATA_ROOT / "processed_dataset" / "mel_tensors"

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


class MusicVAEDataset(Dataset):
    """Loads processed feature rows and row-aligned mel tensor chunks."""

    def __init__(
        self,
        model_input_path: str | Path = DEFAULT_MODEL_INPUT_PATH,
        mel_tensor_dir: str | Path = DEFAULT_MEL_TENSOR_DIR,
        mel_cache_chunks: int = 2,
        indices: list[int] | tuple[int, ...] | torch.Tensor | None = None,
    ) -> None:
        self.model_input_path = Path(model_input_path)
        self.mel_tensor_dir = Path(mel_tensor_dir)
        self.mel_cache_chunks = mel_cache_chunks

        self.features = pq.read_table(self.model_input_path, columns=MODEL_INPUT_COLUMNS)
        self._columns = {name: self.features[name] for name in MODEL_INPUT_COLUMNS}

        metadata_path = self.mel_tensor_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing mel tensor metadata: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.mel_chunk_size = int(metadata["chunk_size"])
        self._mel_chunks: list[dict[str, Any]] = metadata["chunks"]
        self.mel_row_count = int(metadata["row_count"])
        self.full_length = min(self.features.num_rows, self.mel_row_count)

        if indices is None:
            self.indices = list(range(self.full_length))
        else:
            self.indices = [int(index) for index in indices]
            bad = [index for index in self.indices if index < 0 or index >= self.full_length]
            if bad:
                raise IndexError(f"Dataset indices out of range; first invalid index: {bad[0]}")

        self.length = len(self.indices)
        self._mel_cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return self.length

    def _feature_value(self, column: str, index: int) -> Any:
        return self._columns[column][index].as_py()

    def _load_mel_chunk(self, chunk_index: int) -> torch.Tensor:
        chunk = self._mel_chunks[chunk_index]
        return torch.load(chunk["path"], map_location="cpu", weights_only=True)

    def _mel_value(self, index: int) -> torch.Tensor:
        chunk_index = index // self.mel_chunk_size
        offset = index % self.mel_chunk_size

        if chunk_index not in self._mel_cache:
            self._mel_cache[chunk_index] = self._load_mel_chunk(chunk_index)
            self._mel_cache.move_to_end(chunk_index)
            while len(self._mel_cache) > self.mel_cache_chunks:
                self._mel_cache.popitem(last=False)
        else:
            self._mel_cache.move_to_end(chunk_index)

        return self._mel_cache[chunk_index][offset]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = self.indices[index]
        mel = self._mel_value(row_index).float()

        return {
            "row_index": row_index,
            "music_id": self._feature_value("music_id", row_index),
            "music_name": self._feature_value("music_name", row_index),
            "art_id": self._feature_value("art_id", row_index),
            "art_name": self._feature_value("art_name", row_index),
            "lyrics_e5_large_embedding": _as_float_tensor(
                self._feature_value("lyrics_e5_large_embedding", row_index)
            ),
            "melspectrogram": mel,
            "main_genre_embedding": _as_float_tensor(
                self._feature_value("main_genre_embedding", row_index)
            ),
            "music_lang_embedding": _as_float_tensor(
                self._feature_value("music_lang_embedding", row_index)
            ),
            "genre_condition_embedding": _as_float_tensor(
                self._feature_value("genre_condition_embedding", row_index)
            ),
            "language_condition_embedding": _as_float_tensor(
                self._feature_value("language_condition_embedding", row_index)
            ),
            "condition_embedding": _as_float_tensor(
                self._feature_value("condition_embedding", row_index)
            ),
        }

    def close(self) -> None:
        self._mel_cache.clear()


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
    output["row_index"] = torch.tensor([item["row_index"] for item in batch], dtype=torch.long)
    for key in text_keys:
        output[key] = [item[key] for item in batch]
    return output


def create_dataloader(
    model_input_path: str | Path = DEFAULT_MODEL_INPUT_PATH,
    mel_tensor_dir: str | Path = DEFAULT_MEL_TENSOR_DIR,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    mel_cache_chunks: int = 2,
    pin_memory: bool = False,
    indices: list[int] | tuple[int, ...] | torch.Tensor | None = None,
) -> DataLoader:
    dataset = MusicVAEDataset(
        model_input_path=model_input_path,
        mel_tensor_dir=mel_tensor_dir,
        mel_cache_chunks=mel_cache_chunks,
        indices=indices,
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
    loader = create_dataloader(batch_size=2)
    batch = next(iter(loader))
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key}: {value[:2]}")
