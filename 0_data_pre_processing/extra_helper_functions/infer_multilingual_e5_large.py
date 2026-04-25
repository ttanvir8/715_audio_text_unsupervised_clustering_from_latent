import argparse
import json
import logging
import re
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DATA_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATASET = DATA_ROOT / "raw_dataset" / "4mula_small.parquet"
DEFAULT_MODEL_DIR = DATA_ROOT / "embedding_models" / "intfloat" / "multilingual-e5-large"
LOGGER = logging.getLogger(__name__)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def check_local_model(model_dir: Path) -> None:
    required_files = ("config.json", "tokenizer_config.json")
    if model_dir.exists() and all((model_dir / name).exists() for name in required_files):
        return

    raise FileNotFoundError(
        "Local multilingual-e5-large model is missing.\n"
        "Download it from inside 0_data_pre_processing with:\n"
        "hf download intfloat/multilingual-e5-large "
        "--local-dir ./embedding_models/intfloat/multilingual-e5-large"
    )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def load_datapoint(raw_dataset: Path, row_index: int) -> dict[str, object]:
    if row_index < 0:
        raise ValueError("--row-index must be greater than or equal to 0")

    columns = ["music_id", "music_name", "music_lang", "music_lyrics"]
    row = pl.scan_parquet(raw_dataset).select(columns).slice(row_index, 1).collect()
    if row.is_empty():
        raise IndexError(f"No datapoint exists at row index {row_index}")

    datapoint = row.to_dicts()[0]
    datapoint["clean_lyrics"] = clean_text(datapoint.get("music_lyrics"))
    return datapoint


def infer_embedding(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    e5_input = f"passage: {text}" if text else "passage: "
    tokenized = tokenizer(
        [e5_input],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**tokenized)
        pooled = mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
        return F.normalize(pooled, p=2, dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer one raw dataset datapoint with the local intfloat/multilingual-e5-large model."
    )
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    check_local_model(args.model_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    LOGGER.info("Using device: %s", device)

    datapoint = load_datapoint(args.raw_dataset, args.row_index)
    LOGGER.info("Loading tokenizer and model from: %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(device).eval()

    LOGGER.info("Running inference for row index: %s", args.row_index)
    embedding = infer_embedding(
        text=str(datapoint["clean_lyrics"]),
        tokenizer=tokenizer,
        model=model,
        max_length=args.max_length,
        device=device,
    )
    embedding_list = embedding.squeeze(0).cpu().tolist()

    result = {
        "row_index": args.row_index,
        "music_id": datapoint.get("music_id"),
        "music_name": datapoint.get("music_name"),
        "music_lang": datapoint.get("music_lang"),
        "model_dir": str(args.model_dir),
        "device": str(device),
        "input_prefix": "passage: ",
        "clean_lyrics_preview": str(datapoint["clean_lyrics"])[:300],
        "embedding_shape": list(embedding.shape),
        "embedding": embedding_list,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
