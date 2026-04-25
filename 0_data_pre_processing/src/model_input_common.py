import json
from pathlib import Path

import polars as pl


DATA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATASET = DATA_ROOT / "raw_dataset" / "4mula_small.parquet"
DEFAULT_LYRICS_DATASET = DATA_ROOT / "processed_dataset" / "lyrics_e5_large" / "lyrics_embeddings.parquet"
DEFAULT_GENRE_LANGUAGE_DATASET = (
    DATA_ROOT / "processed_dataset" / "genre_language" / "genre_language_embeddings.parquet"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "processed_dataset" / "model_input"

ROW_INDEX_COLUMN = "__row_index"
RAW_COLUMNS = ["music_id", "music_name", "art_id", "art_name"]
LYRICS_COLUMNS = ["music_id", "lyrics_e5_large_embedding"]
GENRE_LANGUAGE_COLUMNS = [
    "music_id",
    "main_genre_embedding",
    "music_lang_embedding",
    "genre_condition_embedding",
    "language_condition_embedding",
    "condition_embedding",
]
FINAL_COLUMNS = [
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


def ensure_unique_ids(df: pl.DataFrame, id_column: str, dataset_name: str) -> None:
    duplicate_count = df.height - df.select(id_column).unique().height
    if duplicate_count:
        raise ValueError(f"{dataset_name} has {duplicate_count} duplicate {id_column} values")


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def chunk_path(output_dir: Path, name: str, index: int) -> Path:
    return output_dir / name / f"chunk_{index:05d}.parquet"
