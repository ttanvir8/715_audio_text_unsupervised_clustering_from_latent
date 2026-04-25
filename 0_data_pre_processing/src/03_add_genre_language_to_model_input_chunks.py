import argparse
import logging
from pathlib import Path

import polars as pl

from model_input_common import (
    DEFAULT_GENRE_LANGUAGE_DATASET,
    DEFAULT_OUTPUT_DIR,
    FINAL_COLUMNS,
    GENRE_LANGUAGE_COLUMNS,
    ROW_INDEX_COLUMN,
    chunk_path,
    read_manifest,
    write_manifest,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add genre/language embeddings and condition vectors to model input chunks."
    )
    parser.add_argument("--genre-language-dataset", type=Path, default=DEFAULT_GENRE_LANGUAGE_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    output_chunk_dir = args.output_dir / "03_final_feature_chunks"
    output_chunk_dir.mkdir(parents=True, exist_ok=True)

    lyrics_manifest = read_manifest(args.output_dir / "02_with_lyrics_chunks_manifest.json")
    genre_language_df = (
        pl.scan_parquet(args.genre_language_dataset)
        .select(GENRE_LANGUAGE_COLUMNS)
        .with_row_index(ROW_INDEX_COLUMN)
        .select([ROW_INDEX_COLUMN, *GENRE_LANGUAGE_COLUMNS[1:]])
        .collect()
    )
    chunk_files = []

    for index, lyrics_path in enumerate(lyrics_manifest["chunk_files"]):
        lyrics_chunk = pl.read_parquet(lyrics_path)
        joined = lyrics_chunk.join(
            genre_language_df, on=ROW_INDEX_COLUMN, how="inner", validate="1:1"
        ).select(FINAL_COLUMNS)
        if joined.height != lyrics_chunk.height:
            raise ValueError(f"Genre/language join lost rows in chunk {index}")

        path = chunk_path(args.output_dir, "03_final_feature_chunks", index)
        joined.write_parquet(path)
        chunk_files.append(str(path))
        LOGGER.info("Wrote final feature chunk %s with %s rows", index, joined.height)

    write_manifest(
        args.output_dir / "03_final_feature_chunks_manifest.json",
        {
            "genre_language_dataset": str(args.genre_language_dataset),
            "row_count": lyrics_manifest["row_count"],
            "chunk_count": len(chunk_files),
            "chunk_files": chunk_files,
            "columns": FINAL_COLUMNS,
        },
    )
    LOGGER.info("Wrote final feature manifest")


if __name__ == "__main__":
    main()
