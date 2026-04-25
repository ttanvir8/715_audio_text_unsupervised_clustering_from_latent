from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


LABEL_COLUMNS = [
    "music_id",
    "main_genre",
    "music_lang",
    "main_genre_id",
    "music_lang_id",
]


def load_label_frame(label_path: str | Path, row_count: int | None = None) -> pd.DataFrame:
    table = pq.read_table(label_path, columns=LABEL_COLUMNS)
    labels = table.to_pandas()
    labels.insert(0, "row_index", np.arange(len(labels), dtype=np.int64))
    if row_count is not None:
        labels = labels.iloc[:row_count].copy()
    return labels


def _valid_stratify(values: pd.Series, val_ratio: float) -> bool:
    counts = values.value_counts()
    if counts.empty or counts.min() < 2:
        return False
    val_size = int(round(len(values) * val_ratio))
    return val_size >= len(counts)


def create_train_val_split(
    labels: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    stratify_by: str = "genre_language",
) -> dict[str, Any]:
    all_indices = labels["row_index"].to_numpy(dtype=np.int64)

    preferred = None
    if stratify_by == "genre_language":
        preferred = labels["main_genre_id"].astype(str) + "_" + labels["music_lang_id"].astype(str)
    elif stratify_by == "genre":
        preferred = labels["main_genre_id"]

    fallback = labels["main_genre_id"]
    stratify = None
    strategy = "none"
    if preferred is not None and _valid_stratify(preferred, val_ratio):
        stratify = preferred
        strategy = stratify_by
    elif _valid_stratify(fallback, val_ratio):
        stratify = fallback
        strategy = "genre"

    test_size = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    return {
        "train_indices": sorted(int(index) for index in train_indices),
        "val_indices": sorted(int(index) for index in val_indices),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "stratify_by_requested": stratify_by,
        "stratify_by_used": strategy,
        "row_count": int(len(labels)),
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
    val_index_set = pd.Index(val_indices)
    val_labels = labels.set_index("row_index").loc[val_index_set].reset_index()
    val_labels = val_labels.rename(columns={"index": "row_index"})
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(val_labels, preserve_index=False), path)
    return val_labels
