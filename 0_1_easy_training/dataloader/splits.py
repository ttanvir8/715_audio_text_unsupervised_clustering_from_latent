from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


LABEL_COLUMNS = [
    "music_id",
    "music_name",
    "music_lang",
]


def load_label_frame(
    lyrics_path: str | Path,
    language_column: str = "music_lang",
) -> pd.DataFrame:
    columns = list(dict.fromkeys([*LABEL_COLUMNS, language_column]))
    table = pq.read_table(lyrics_path, columns=columns)
    labels = table.to_pandas()
    labels.insert(0, "row_index", np.arange(len(labels), dtype=np.int64))
    return labels


def _split_count(row_count: int, train_ratio: float, val_ratio: float) -> int:
    if row_count < 2:
        return 0
    val_fraction = val_ratio / (train_ratio + val_ratio)
    val_count = int(round(row_count * val_fraction))
    val_count = max(1, val_count)
    return min(row_count - 1, val_count)


def create_language_ratio_split(
    labels: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    language_column: str = "music_lang",
    max_rows_per_language: int | None = None,
    min_rows_per_language: int = 2,
) -> dict[str, Any]:
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must both be positive.")
    if language_column not in labels.columns:
        raise KeyError(f"Missing language column '{language_column}' in labels.")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    selected_indices: list[int] = []
    skipped_languages: dict[str, int] = {}
    per_language: dict[str, dict[str, int]] = {}

    grouped = labels.groupby(language_column, dropna=False, sort=True)
    for language, frame in grouped:
        language_name = str(language)
        row_indices = frame["row_index"].to_numpy(dtype=np.int64)
        rng.shuffle(row_indices)

        if max_rows_per_language is not None and max_rows_per_language > 0:
            row_indices = row_indices[:max_rows_per_language]

        if len(row_indices) < min_rows_per_language:
            skipped_languages[language_name] = int(len(row_indices))
            continue

        val_count = _split_count(len(row_indices), train_ratio, val_ratio)
        val_group = row_indices[:val_count]
        train_group = row_indices[val_count:]

        train_indices.extend(int(index) for index in train_group)
        val_indices.extend(int(index) for index in val_group)
        selected_indices.extend(int(index) for index in row_indices)
        per_language[language_name] = {
            "selected": int(len(row_indices)),
            "train": int(len(train_group)),
            "val": int(len(val_group)),
        }

    if not train_indices or not val_indices:
        raise ValueError("The language split produced an empty train or validation split.")

    return {
        "train_indices": sorted(train_indices),
        "val_indices": sorted(val_indices),
        "selected_indices": sorted(selected_indices),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "language_column": language_column,
        "max_rows_per_language": max_rows_per_language,
        "min_rows_per_language": int(min_rows_per_language),
        "split_strategy": "per_language_same_ratio",
        "per_language": per_language,
        "skipped_languages": skipped_languages,
        "row_count": int(len(selected_indices)),
    }


def save_split(split: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")


def load_split(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_validation_labels(
    labels: pd.DataFrame,
    val_indices: list[int],
    output_path: str | Path,
) -> pd.DataFrame:
    val_index = pd.Index([int(index) for index in val_indices])
    val_labels = labels.set_index("row_index").loc[val_index].reset_index()
    val_labels = val_labels.rename(columns={"index": "row_index"})
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(val_labels, preserve_index=False), path)
    return val_labels

