import argparse
import json
import re
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATASET = PROJECT_ROOT / "raw_dataset" / "4mula_small.parquet"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "embedding_models" / "intfloat" / "multilingual-e5-base"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed_dataset" / "lyrics"
E5_EMBEDDING_DIM = 768
LYRICS_EMBEDDING_DIM = 256


def clean_lyrics(value: object) -> str:
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
        "Local multilingual-e5-base model is missing.\n"
        "Download it from inside 0_data_pre_processing with:\n"
        "hf download intfloat/multilingual-e5-base "
        "--local-dir ./embedding_models/intfloat/multilingual-e5-base"
    )


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def make_projection(seed: int, device: torch.device) -> nn.Linear:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = nn.Linear(E5_EMBEDDING_DIM, LYRICS_EMBEDDING_DIM)
    with torch.no_grad():
        weights = torch.empty((LYRICS_EMBEDDING_DIM, E5_EMBEDDING_DIM))
        nn.init.xavier_uniform_(weights, generator=generator)
        projection.weight.copy_(weights)
        projection.bias.zero_()
    return projection.to(device).eval()


def load_lyrics(raw_dataset: Path, limit: int | None) -> pl.DataFrame:
    columns = ["music_id", "music_lang", "music_lyrics"]
    query = pl.scan_parquet(raw_dataset).select(columns)
    if limit is not None:
        query = query.head(limit)
    return query.collect().with_columns(
        pl.col("music_lyrics")
        .map_elements(clean_lyrics, return_dtype=pl.String)
        .alias("clean_lyrics")
    )


def encode_batches(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    projection: nn.Linear,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[list[list[float]], list[list[float]]]:
    e5_embeddings: list[list[float]] = []
    projected_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        e5_inputs = [f"passage: {text}" if text else "passage: " for text in batch_texts]

        tokenized = tokenizer(
            e5_inputs,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(**tokenized)
            pooled = mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            projected = projection(pooled)

        e5_embeddings.extend(pooled.cpu().tolist())
        projected_embeddings.extend(projected.cpu().tolist())

        print(f"encoded {min(start + batch_size, len(texts))}/{len(texts)} lyrics")

    return e5_embeddings, projected_embeddings


def write_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    metadata = {
        "raw_dataset": str(args.raw_dataset),
        "model_dir": str(args.model_dir),
        "text_cleaning": "normalize whitespace, keep original language, no stemming",
        "e5_prefix": "passage: ",
        "e5_embedding_dim": E5_EMBEDDING_DIM,
        "lyrics_embedding_dim": LYRICS_EMBEDDING_DIM,
        "max_length": args.max_length,
        "projection_seed": args.projection_seed,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess music_lyrics into multilingual-e5-base embeddings and 256-d projected vectors."
    )
    parser.add_argument("--raw-dataset", type=Path, default=DEFAULT_RAW_DATASET)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--projection-seed", type=int, default=751)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_local_model(args.model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    df = load_lyrics(args.raw_dataset, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(device).eval()
    projection = make_projection(args.projection_seed, device)

    e5_embeddings, lyrics_embeddings = encode_batches(
        texts=df["clean_lyrics"].to_list(),
        tokenizer=tokenizer,
        model=model,
        projection=projection,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    output = df.select(["music_id", "music_lang", "clean_lyrics"]).with_columns(
        pl.Series("lyrics_e5_embedding", e5_embeddings),
        pl.Series("lyrics_embedding", lyrics_embeddings),
    )

    output_path = args.output_dir / "lyrics_embeddings.parquet"
    projection_path = args.output_dir / "lyrics_projection.pt"
    output.write_parquet(output_path)
    torch.save(projection.state_dict(), projection_path)
    write_metadata(args.output_dir, args)

    print(f"wrote {output.height} rows to {output_path}")
    print(f"wrote projection weights to {projection_path}")


if __name__ == "__main__":
    main()
