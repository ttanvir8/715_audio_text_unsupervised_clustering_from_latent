import argparse
import json
import logging
from pathlib import Path

import duckdb
import torch.nn.functional as F
import torch


DATA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATASET = DATA_ROOT / "raw_dataset" / "4mula_small.parquet"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "processed_dataset" / "mel_tensors"
LOGGER = logging.getLogger(__name__)
TARGET_MEL_BINS = 128
TARGET_MEL_FRAMES = 1292


def quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def normalize_mel_shape(mel: object, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.tensor(mel, dtype=dtype)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D mel tensor, got shape {tuple(tensor.shape)}")

    mel_bins, frames = tensor.shape
    if mel_bins < TARGET_MEL_BINS:
        tensor = F.pad(tensor, (0, 0, 0, TARGET_MEL_BINS - mel_bins))
    elif mel_bins > TARGET_MEL_BINS:
        tensor = tensor[:TARGET_MEL_BINS, :]

    if frames < TARGET_MEL_FRAMES:
        tensor = F.pad(tensor, (0, TARGET_MEL_FRAMES - frames))
    elif frames > TARGET_MEL_FRAMES:
        tensor = tensor[:, :TARGET_MEL_FRAMES]

    return tensor.unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract raw melspectrogram values into chunked torch tensor files."
    )
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--memory-limit", default="12GB")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tensor_dtype = dtype_from_name(args.dtype)
    con = duckdb.connect()
    con.execute(f"set memory_limit='{args.memory_limit}'")
    con.execute("set threads=1")
    con.execute("set preserve_insertion_order=false")

    total_rows = con.execute(
        f"select count(*) from read_parquet('{quote_path(args.raw_dataset)}')"
    ).fetchone()[0]
    if args.limit is not None:
        total_rows = min(total_rows, args.limit)

    chunks = []
    LOGGER.info(
        "Extracting %s mel rows into %s-row %s tensor chunks",
        total_rows,
        args.chunk_size,
        args.dtype,
    )

    start_chunk = 0
    if args.resume:
        existing_chunks = sorted(args.output_dir.glob("chunk_*.pt"))
        start_chunk = len(existing_chunks)
        chunks.extend(
            {
                "chunk_index": index,
                "path": str(path),
                "start": index * args.chunk_size,
                "rows": int(torch.load(path, map_location="cpu", weights_only=True).shape[0]),
                "shape": list(torch.load(path, map_location="cpu", weights_only=True).shape),
            }
            for index, path in enumerate(existing_chunks)
        )
        LOGGER.info("Resuming after %s existing chunks", start_chunk)

    for chunk_index, start in enumerate(
        range(start_chunk * args.chunk_size, total_rows, args.chunk_size),
        start=start_chunk,
    ):
        limit = min(args.chunk_size, total_rows - start)
        query = f"""
            select melspectrogram
            from read_parquet('{quote_path(args.raw_dataset)}')
            limit {limit}
            offset {start}
        """
        rows = con.execute(query).fetchall()
        tensors = [normalize_mel_shape(row[0], tensor_dtype) for row in rows]
        chunk_tensor = torch.stack(tensors, dim=0).contiguous()

        chunk_path = args.output_dir / f"chunk_{chunk_index:05d}.pt"
        torch.save(chunk_tensor, chunk_path)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "path": str(chunk_path),
                "start": start,
                "rows": int(chunk_tensor.shape[0]),
                "shape": list(chunk_tensor.shape),
            }
        )
        LOGGER.info("Wrote %s with shape %s", chunk_path, tuple(chunk_tensor.shape))

    metadata = {
        "raw_dataset": str(args.raw_dataset),
        "row_count": total_rows,
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "dtype": args.dtype,
        "tensor_shape_per_row": [1, TARGET_MEL_BINS, TARGET_MEL_FRAMES],
        "shape_policy": "pad or crop every mel to [1, 128, 1292]",
        "chunks": chunks,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Wrote metadata to %s", args.output_dir / "metadata.json")


if __name__ == "__main__":
    main()
