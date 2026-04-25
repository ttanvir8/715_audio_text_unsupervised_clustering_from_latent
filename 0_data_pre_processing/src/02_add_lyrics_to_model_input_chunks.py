import argparse
import logging
from pathlib import Path

import polars as pl

from model_input_common import (
    DEFAULT_LYRICS_DATASET,
    DEFAULT_OUTPUT_DIR,
    LYRICS_COLUMNS,
    RAW_COLUMNS,
    ROW_INDEX_COLUMN,
    chunk_path,
    read_manifest,
    write_manifest,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add lyrics_e5_large_embedding to base chunks.")
    parser.add_argument("--lyrics-dataset", type=Path, default=DEFAULT_LYRICS_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    output_chunk_dir = args.output_dir / "02_with_lyrics_chunks"
    output_chunk_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = read_manifest(args.output_dir / "01_base_chunks_manifest.json")
    lyrics_df = (
        pl.scan_parquet(args.lyrics_dataset)
        .select(LYRICS_COLUMNS)
        .with_row_index(ROW_INDEX_COLUMN)
        .select([ROW_INDEX_COLUMN, "lyrics_e5_large_embedding"])
        .collect()
    )

    output_columns = [
        ROW_INDEX_COLUMN,
        *RAW_COLUMNS[:2],
        "lyrics_e5_large_embedding",
        *RAW_COLUMNS[2:],
    ]
    chunk_files = []

    for index, base_path in enumerate(base_manifest["chunk_files"]):
        base_chunk = pl.read_parquet(base_path)
        joined = base_chunk.join(
            lyrics_df, on=ROW_INDEX_COLUMN, how="inner", validate="1:1"
        ).select(output_columns)
        if joined.height != base_chunk.height:
            raise ValueError(f"Lyrics join lost rows in chunk {index}")

        path = chunk_path(args.output_dir, "02_with_lyrics_chunks", index)
        joined.write_parquet(path)
        chunk_files.append(str(path))
        LOGGER.info("Wrote lyrics chunk %s with %s rows", index, joined.height)

    write_manifest(
        args.output_dir / "02_with_lyrics_chunks_manifest.json",
        {
            "lyrics_dataset": str(args.lyrics_dataset),
            "row_count": base_manifest["row_count"],
            "chunk_count": len(chunk_files),
            "chunk_files": chunk_files,
            "columns": output_columns,
        },
    )
    LOGGER.info("Wrote lyrics manifest")


if __name__ == "__main__":
    main()
