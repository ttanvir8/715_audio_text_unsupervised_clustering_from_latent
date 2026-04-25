import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from model_input_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RAW_DATASET,
    RAW_COLUMNS,
    ROW_INDEX_COLUMN,
    chunk_path,
    write_manifest,
)


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create raw base chunks for the final model input dataset."
    )
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    base_dir = args.output_dir / "01_base_chunks"
    base_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(args.raw_dataset)
    row_count = parquet_file.metadata.num_rows
    chunk_files = []
    LOGGER.info("Creating raw chunks from %s rows with chunk size %s", row_count, args.chunk_size)

    offset = 0
    for index, batch in enumerate(
        parquet_file.iter_batches(batch_size=args.chunk_size, columns=RAW_COLUMNS)
    ):
        table = pa.Table.from_batches([batch])
        row_index = pa.array(range(offset, offset + table.num_rows), type=pa.int64())
        table = table.add_column(0, ROW_INDEX_COLUMN, row_index)
        offset += table.num_rows
        path = chunk_path(args.output_dir, "01_base_chunks", index)
        pq.write_table(table, path)
        chunk_files.append(str(path))
        LOGGER.info("Wrote base chunk %s with %s rows", index, table.num_rows)

    write_manifest(
        args.output_dir / "01_base_chunks_manifest.json",
        {
            "raw_dataset": str(args.raw_dataset),
            "row_count": row_count,
            "chunk_size": args.chunk_size,
            "chunk_count": len(chunk_files),
            "chunk_files": chunk_files,
            "columns": [ROW_INDEX_COLUMN, *RAW_COLUMNS],
        },
    )
    LOGGER.info("Wrote base manifest")


if __name__ == "__main__":
    main()
