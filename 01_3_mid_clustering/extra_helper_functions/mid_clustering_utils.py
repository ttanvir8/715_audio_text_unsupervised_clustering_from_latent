from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_int_csv_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_path)
    else:
        df.to_csv(output_path, index=False)


def load_latent_scope(scope_dir: str | Path, embedding: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    path = Path(scope_dir)
    values = np.load(path / f"{embedding}.npy").astype(np.float32, copy=False)
    row_indices = np.load(path / "row_indices.npy").astype(np.int64, copy=False)
    labels = pq.read_table(path / "labels.parquet").to_pandas()

    if values.shape[0] != len(row_indices) or values.shape[0] != len(labels):
        raise ValueError(
            f"Latent rows, row indices, and labels must align in {path}: "
            f"{values.shape[0]}, {len(row_indices)}, {len(labels)}"
        )
    if "row_index" in labels.columns and not np.array_equal(labels["row_index"].to_numpy(), row_indices):
        raise ValueError(f"labels.parquet row_index does not match row_indices.npy in {path}")
    return values, row_indices, labels


def standardized(values: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(values).astype(np.float32, copy=False)


def fit_kmeans(values: np.ndarray, k: int, seed: int, n_init: int) -> np.ndarray:
    if k < 2:
        raise ValueError("k must be at least 2.")
    if k > values.shape[0]:
        raise ValueError(f"k={k} exceeds sample count={values.shape[0]}.")
    model = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    return model.fit_predict(values).astype(np.int64, copy=False)


def safe_silhouette(values: np.ndarray, clusters: np.ndarray) -> float:
    unique_count = len(np.unique(clusters))
    if unique_count < 2 or unique_count >= values.shape[0]:
        return float("nan")
    return float(silhouette_score(values, clusters))


def safe_calinski_harabasz(values: np.ndarray, clusters: np.ndarray) -> float:
    unique_count = len(np.unique(clusters))
    if unique_count < 2 or unique_count >= values.shape[0]:
        return float("nan")
    return float(calinski_harabasz_score(values, clusters))


def safe_davies_bouldin(values: np.ndarray, clusters: np.ndarray) -> float:
    unique_count = len(np.unique(clusters))
    if unique_count < 2 or unique_count >= values.shape[0]:
        return float("nan")
    return float(davies_bouldin_score(values, clusters))


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    contingency = pd.crosstab(pd.Series(y_pred, name="cluster"), pd.Series(y_true, name="target"))
    if contingency.empty:
        return float("nan")
    return float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())


def supervised_metrics(values: np.ndarray, clusters: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "silhouette": safe_silhouette(values, clusters),
        "nmi": float(normalized_mutual_info_score(target, clusters)),
        "ari": float(adjusted_rand_score(target, clusters)),
        "purity": purity_score(target, clusters),
    }


def named_cluster_summary(
    assignments: pd.DataFrame,
    assignment_column: str,
    label_column: str,
    label_slug: str,
) -> pd.DataFrame:
    rows = []
    for cluster_id, group in assignments.groupby(assignment_column, sort=True):
        counts = group[label_column].value_counts(dropna=False)
        top_label = counts.index[0] if not counts.empty else None
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        rows.append(
            {
                "assignment": assignment_column,
                "cluster": int(cluster_id),
                "size": int(len(group)),
                f"top_{label_slug}": top_label,
                f"top_{label_slug}_count": top_count,
                f"top_{label_slug}_fraction": float(top_count / len(group)) if len(group) else float("nan"),
                f"unique_{label_slug}s": int(group[label_column].nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows)


def observed_labels(
    labels: pd.DataFrame,
    label_column: str,
    label_id_column: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    if label_id_column and label_id_column in labels.columns:
        label_ids = labels[label_id_column].to_numpy(dtype=np.int64, copy=False)
        ordered_names = (
            labels[[label_id_column, label_column]]
            .drop_duplicates()
            .sort_values(label_id_column)[label_column]
            .astype(str)
            .tolist()
        )
        return label_ids, ordered_names

    codes, names = pd.factorize(labels[label_column], sort=True)
    return codes.astype(np.int64), [str(name) for name in names.tolist()]


def load_vector_rows(
    parquet_path: str | Path,
    column: str,
    row_indices: np.ndarray,
) -> np.ndarray:
    table = pq.read_table(parquet_path, columns=[column])
    selected = table.take(pa.array(row_indices, type=pa.int64()))
    values = np.asarray(selected[column].to_pylist(), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array from {column}, got {values.shape}")
    return values


def pca_embedding(values: np.ndarray, components: int, seed: int) -> tuple[np.ndarray, PCA]:
    component_count = min(int(components), values.shape[0], values.shape[1])
    if component_count < 1:
        raise ValueError("pca-components must be at least 1.")
    model = PCA(n_components=component_count, random_state=seed)
    features = model.fit_transform(standardized(values)).astype(np.float32, copy=False)
    return features, model


def _mel_feature_block(mel_batch) -> np.ndarray:
    import torch

    mel_log = torch.log1p(mel_batch.float()).squeeze(1)
    freq_mean = mel_log.mean(dim=2)
    freq_std = mel_log.std(dim=2, unbiased=False)
    global_stats = torch.stack(
        [
            mel_log.mean(dim=(1, 2)),
            mel_log.std(dim=(1, 2), unbiased=False),
            mel_log.amin(dim=(1, 2)),
            mel_log.amax(dim=(1, 2)),
        ],
        dim=1,
    )
    return torch.cat([freq_mean, freq_std, global_stats], dim=1).cpu().numpy()


def _pool_time_features(mel_batch, time_bins: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if time_bins <= 0:
        return np.empty((mel_batch.shape[0], 0), dtype=np.float32)
    mel_log = torch.log1p(mel_batch.float()).squeeze(1)
    time_mean = mel_log.mean(dim=1).unsqueeze(1)
    time_std = mel_log.std(dim=1, unbiased=False).unsqueeze(1)
    pooled_mean = F.adaptive_avg_pool1d(time_mean, time_bins).squeeze(1)
    pooled_std = F.adaptive_avg_pool1d(time_std, time_bins).squeeze(1)
    return torch.cat([pooled_mean, pooled_std], dim=1).cpu().numpy()


def load_spectral_features(
    mel_tensor_dir: str | Path,
    row_indices: np.ndarray,
    time_bins: int = 32,
) -> np.ndarray:
    import torch

    mel_dir = Path(mel_tensor_dir)
    metadata = json.loads((mel_dir / "metadata.json").read_text(encoding="utf-8"))
    chunk_size = int(metadata["chunk_size"])
    chunks = metadata["chunks"]
    row_indices = np.asarray(row_indices, dtype=np.int64)
    output: np.ndarray | None = None

    by_chunk: dict[int, list[tuple[int, int]]] = {}
    for output_index, row_index in enumerate(row_indices.tolist()):
        chunk_index = int(row_index // chunk_size)
        offset = int(row_index % chunk_size)
        by_chunk.setdefault(chunk_index, []).append((output_index, offset))

    for chunk_index, entries in sorted(by_chunk.items()):
        chunk = chunks[chunk_index]
        tensor = torch.load(chunk["path"], map_location="cpu", weights_only=True)
        offsets = [offset for _, offset in entries]
        mel_batch = tensor[offsets]
        features = np.concatenate(
            [
                _mel_feature_block(mel_batch),
                _pool_time_features(mel_batch, time_bins),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        if output is None:
            output = np.empty((len(row_indices), features.shape[1]), dtype=np.float32)
        for feature_index, (output_index, _) in enumerate(entries):
            output[output_index] = features[feature_index]

    if output is None:
        raise ValueError("No spectral features were loaded.")
    return output


def audio_lyrics_raw_features(
    model_input_path: str | Path,
    mel_tensor_dir: str | Path,
    row_indices: np.ndarray,
    lyrics_embedding_column: str,
    spectral_time_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lyrics = load_vector_rows(model_input_path, lyrics_embedding_column, row_indices)
    spectral = load_spectral_features(mel_tensor_dir, row_indices, time_bins=spectral_time_bins)
    combined = np.concatenate([lyrics, spectral], axis=1).astype(np.float32, copy=False)
    return combined, lyrics, spectral


def autoencoder_embedding(
    values: np.ndarray,
    latent_dim: int = 64,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 751,
    device_name: str = "auto",
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Autoencoder CUDA device was requested, but CUDA is not available.")

    x_np = standardized(values).astype(np.float32, copy=False)
    input_dim = int(x_np.shape[1])
    latent_dim = min(int(latent_dim), input_dim)
    hidden_dim = min(512, max(128, latent_dim * 4))

    class FeatureAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    model = FeatureAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_np)),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )

    model.train()
    for _ in range(max(1, int(epochs))):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, x_np.shape[0], int(batch_size)):
            batch = torch.from_numpy(x_np[start : start + int(batch_size)]).to(device)
            chunks.append(model.encoder(batch).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
