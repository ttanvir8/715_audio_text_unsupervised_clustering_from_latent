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
    "main_genre",
    "music_lang",
    "main_genre_id",
    "music_lang_id",
]

MODEL_METADATA_COLUMNS = [
    "music_id",
    "music_name",
    "art_id",
    "art_name",
]


def _available_columns(path: str | Path, requested: list[str]) -> list[str]:
    schema_names = set(pq.read_schema(path).names)
    return [column for column in requested if column in schema_names]


def load_label_frame(
    label_path: str | Path,
    language_column: str = "music_lang",
    row_count: int | None = None,
    model_input_path: str | Path | None = None,
) -> pd.DataFrame:
    columns = list(dict.fromkeys([*LABEL_COLUMNS, language_column]))
    columns = _available_columns(label_path, columns)
    table = pq.read_table(label_path, columns=columns)
    labels = table.to_pandas()

    if row_count is not None:
        labels = labels.iloc[:row_count].copy()

    if model_input_path is not None:
        metadata_columns = _available_columns(model_input_path, MODEL_METADATA_COLUMNS)
        metadata = pq.read_table(model_input_path, columns=metadata_columns).to_pandas()
        if row_count is not None:
            metadata = metadata.iloc[:row_count].copy()
        if "music_id" in labels.columns and "music_id" in metadata.columns:
            same_ids = labels["music_id"].astype(str).to_numpy() == metadata["music_id"].astype(str).to_numpy()
            if not bool(np.all(same_ids)):
                raise ValueError("label_path and model_input_path are not row-aligned by music_id.")
        for column in metadata.columns:
            if column not in labels.columns:
                labels[column] = metadata[column].to_numpy()

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

