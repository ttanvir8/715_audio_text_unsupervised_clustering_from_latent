import argparse
import logging
import subprocess
import sys
from pathlib import Path

from model_input_common import DEFAULT_OUTPUT_DIR


SCRIPT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def run_step(script_name: str, output_dir: Path, extra_args: list[str]) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / script_name),
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]
    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the chunked model input dataset build pipeline.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()
    run_step(
        "01_create_model_input_base_chunks.py",
        args.output_dir,
        ["--chunk-size", str(args.chunk_size)],
    )
    run_step("02_add_lyrics_to_model_input_chunks.py", args.output_dir, [])
    run_step("03_add_genre_language_to_model_input_chunks.py", args.output_dir, [])
    run_step("04_compact_model_input_chunks.py", args.output_dir, [])


if __name__ == "__main__":
    main()
