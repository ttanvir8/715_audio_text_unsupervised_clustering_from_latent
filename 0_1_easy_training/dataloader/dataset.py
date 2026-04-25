from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "0_data_pre_processing"
DEFAULT_LYRICS_PATH = (
    DATA_ROOT / "processed_dataset" / "lyrics_e5_large" / "lyrics_embeddings.parquet"
)

BASE_COLUMNS = [
    "music_id",
    "music_name",
    "music_lang",
    "clean_lyrics",
]


def _as_float_tensor(value: Any) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


class LyricsVAEDataset(Dataset):
    """Loads row-aligned lyrics embeddings for a lyrics-only VAE."""

    def __init__(
        self,
        lyrics_path: str | Path = DEFAULT_LYRICS_PATH,
        input_embedding_column: str = "lyrics_e5_large_embedding",
        indices: list[int] | tuple[int, ...] | torch.Tensor | None = None,
        include_clean_lyrics: bool = False,
    ) -> None:
        self.lyrics_path = Path(lyrics_path)
        self.input_embedding_column = input_embedding_column
        self.include_clean_lyrics = include_clean_lyrics

        columns = [column for column in BASE_COLUMNS if include_clean_lyrics or column != "clean_lyrics"]
        columns.append(input_embedding_column)
        self.features = pq.read_table(self.lyrics_path, columns=columns)
        self._columns = {name: self.features[name] for name in columns}
        self.full_length = self.features.num_rows

        if indices is None:
            self.indices = list(range(self.full_length))
        else:
            self.indices = [int(index) for index in indices]
            bad = [index for index in self.indices if index < 0 or index >= self.full_length]
            if bad:
                raise IndexError(f"Dataset indices out of range; first invalid index: {bad[0]}")

        self.length = len(self.indices)
        if self.length == 0:
            raise ValueError("LyricsVAEDataset received no rows.")
        self.input_dim = len(self._feature_value(input_embedding_column, self.indices[0]))

    def __len__(self) -> int:
        return self.length

    def _feature_value(self, column: str, index: int) -> Any:
        return self._columns[column][index].as_py()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = self.indices[index]
        item = {
            "row_index": row_index,
            "music_id": self._feature_value("music_id", row_index),
            "music_name": self._feature_value("music_name", row_index),
            "music_lang": self._feature_value("music_lang", row_index),
            "lyrics_input": _as_float_tensor(
                self._feature_value(self.input_embedding_column, row_index)
            ),
        }
        if self.include_clean_lyrics and "clean_lyrics" in self._columns:
            item["clean_lyrics"] = self._feature_value("clean_lyrics", row_index)
        return item


def lyrics_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "lyrics_input": torch.stack([item["lyrics_input"] for item in batch], dim=0),
        "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        "music_id": [item["music_id"] for item in batch],
        "music_name": [item["music_name"] for item in batch],
        "music_lang": [item["music_lang"] for item in batch],
    }
    if "clean_lyrics" in batch[0]:
        output["clean_lyrics"] = [item["clean_lyrics"] for item in batch]
    return output


def create_dataloader(
    lyrics_path: str | Path = DEFAULT_LYRICS_PATH,
    input_embedding_column: str = "lyrics_e5_large_embedding",
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    indices: list[int] | tuple[int, ...] | torch.Tensor | None = None,
    include_clean_lyrics: bool = False,
) -> DataLoader:
    dataset = LyricsVAEDataset(
        lyrics_path=lyrics_path,
        input_embedding_column=input_embedding_column,
        indices=indices,
        include_clean_lyrics=include_clean_lyrics,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lyrics_collate_fn,
    )


if __name__ == "__main__":
    loader = create_dataloader(batch_size=2)
    batch = next(iter(loader))
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key}: {value[:2]}")

