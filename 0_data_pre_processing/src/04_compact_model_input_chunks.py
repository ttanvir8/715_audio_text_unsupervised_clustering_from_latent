import argparse
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq

from model_input_common import DEFAULT_OUTPUT_DIR, FINAL_COLUMNS, read_manifest


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact final model input chunks into one parquet file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", default="model_input_dataset.parquet")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    manifest = read_manifest(args.output_dir / "03_final_feature_chunks_manifest.json")
    output_path = args.output_dir / args.output_file

    writer = None
    total_rows = 0
    try:
        for index, chunk_file in enumerate(manifest["chunk_files"]):
            table = pq.read_table(chunk_file, columns=FINAL_COLUMNS)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += table.num_rows
            LOGGER.info("Appended chunk %s with %s rows", index, table.num_rows)
    finally:
        if writer is not None:
            writer.close()

    if total_rows != manifest["row_count"]:
        raise ValueError(f"Final dataset has {total_rows} rows, expected {manifest['row_count']}")

    metadata = {
        "row_count": total_rows,
        "output_file": args.output_file,
        "source_manifest": "03_final_feature_chunks_manifest.json",
        "columns": FINAL_COLUMNS,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Wrote compact dataset to %s", output_path)
    LOGGER.info("Wrote metadata to %s", args.output_dir / "metadata.json")


if __name__ == "__main__":
    main()
