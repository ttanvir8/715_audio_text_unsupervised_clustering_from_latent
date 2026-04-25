from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA

from cuda_kmeans import torch_kmeans_predict


def assert_cuml_cuda_available() -> dict[str, int | str]:
    try:
        import cupy
        import cuml
    except ImportError as exc:
        raise RuntimeError(
            "cuml_kmeans requested but RAPIDS cuML is not installed in this Python environment. "
            "Activate the Voxtral uv env and install `cuml-cu12`."
        ) from exc

    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
        runtime_version = int(cupy.cuda.runtime.runtimeGetVersion())
        driver_version = int(cupy.cuda.runtime.driverGetVersion())
    except Exception as exc:
        raise RuntimeError(
            "cuML cannot access CUDA from WSL2. Check `nvidia-smi`, the Windows NVIDIA driver, "
            "and that `/usr/lib/wsl/lib` is visible inside WSL."
        ) from exc

    if device_count < 1:
        raise RuntimeError("cuML cannot access CUDA from WSL2 because no NVIDIA GPU is visible.")

    return {
        "device_count": device_count,
        "cuda_runtime": runtime_version,
        "cuda_driver": driver_version,
        "cuml_version": str(cuml.__version__),
    }


def cuml_device_summary() -> dict[str, int | str]:
    assert_cuml_cuda_available()
    import cupy

    device = cupy.cuda.Device()
    properties = cupy.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return {
        "device_id": int(device.id),
        "device_name": str(name),
        "memory_total_mb": int(properties["totalGlobalMem"] // (1024 * 1024)),
        "compute_capability": f"{properties['major']}.{properties['minor']}",
    }


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


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


def load_latent_scope(scope_dir: str | Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    path = Path(scope_dir)
    embeddings = {
        "mu": np.load(path / "mu.npy"),
        "metadata_assisted": np.load(path / "metadata_assisted.npy"),
    }
    labels = pq.read_table(path / "labels.parquet").to_pandas()
    for name, values in embeddings.items():
        if values.shape[0] != len(labels):
            raise ValueError(
                f"{name} rows ({values.shape[0]}) do not match labels ({len(labels)}) in {path}"
            )
    return embeddings, labels


def standardized(values: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(values)


def pca_embedding(
    values: np.ndarray,
    n_components: int = 64,
    seed: int = 751,
) -> np.ndarray:
    component_count = min(int(n_components), values.shape[0], values.shape[1])
    if component_count < 1:
        raise ValueError(f"Cannot build PCA embedding from shape={values.shape}.")
    model = PCA(
        n_components=component_count,
        svd_solver="randomized" if component_count < min(values.shape) else "auto",
        random_state=seed,
    )
    return model.fit_transform(standardized(values)).astype(np.float32, copy=False)


def _stack_vector_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.asarray(frame[column].tolist(), dtype=np.float32)


def load_model_input_feature_frame(
    model_input_path: str | Path,
    row_indices: np.ndarray,
    vector_columns: list[str],
) -> pd.DataFrame:
    columns = ["music_id", *vector_columns]
    table = pq.read_table(model_input_path, columns=columns)
    frame = table.to_pandas().iloc[row_indices].reset_index(drop=True)
    return frame


def load_tabular_vectors(
    model_input_path: str | Path,
    row_indices: np.ndarray,
    vector_columns: list[str],
) -> np.ndarray:
    frame = load_model_input_feature_frame(model_input_path, row_indices, vector_columns)
    arrays = [_stack_vector_column(frame, column) for column in vector_columns]
    return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)


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


def fit_predict(
    algorithm: str,
    values: np.ndarray,
    k: int | None,
    seed: int,
    cuda_device: str = "cuda",
    cuda_n_init: int = 20,
    cuda_max_iter: int = 100,
    cuda_tol: float = 1e-4,
    cuda_batch_size: int | None = None,
    agglomerative_linkage: str = "ward",
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
) -> np.ndarray:
    if algorithm in {"dbscan", "cuml_dbscan"}:
        k = None
    else:
        if k is None or k < 2:
            raise ValueError("k must be at least 2.")
        if k > values.shape[0]:
            raise ValueError(f"k={k} exceeds sample count={values.shape[0]}.")

    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        return model.fit_predict(values)
    if algorithm == "cuml_kmeans":
        return cuml_kmeans_predict(
            values=values,
            n_clusters=k,
            seed=seed,
            n_init=cuda_n_init,
            max_iter=cuda_max_iter,
            tol=cuda_tol,
        )
    if algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k, linkage=agglomerative_linkage)
        return model.fit_predict(values)
    if algorithm == "cuml_agglomerative":
        return cuml_agglomerative_predict(
            values=values,
            n_clusters=k,
            linkage=agglomerative_linkage,
        )
    if algorithm == "dbscan":
        model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        return model.fit_predict(values)
    if algorithm == "cuml_dbscan":
        return cuml_dbscan_predict(
            values=values,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
        )
    if algorithm == "cuda_kmeans":
        return torch_kmeans_predict(
            values=values,
            n_clusters=k,
            seed=seed,
            device=cuda_device,
            n_init=cuda_n_init,
            max_iter=cuda_max_iter,
            tol=cuda_tol,
            batch_size=cuda_batch_size,
        )
    if algorithm == "gmm":
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-6,
            n_init=5,
            random_state=seed,
        )
        return model.fit_predict(values)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def cuml_kmeans_predict(
    values: np.ndarray,
    n_clusters: int,
    seed: int = 751,
    n_init: int = 20,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> np.ndarray:
    try:
        from cuml.cluster import KMeans as CuMLKMeans
    except ImportError as exc:
        raise RuntimeError(
            "cuml_kmeans requested but RAPIDS cuML is not installed in this Python environment. "
            "Activate the Voxtral uv env and install `cuml-cu12`."
        ) from exc

    assert_cuml_cuda_available()
    model = CuMLKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init=max(1, int(n_init)),
        max_iter=max_iter,
        tol=tol,
        output_type="numpy",
    )
    try:
        labels = model.fit_predict(np.asarray(values, dtype=np.float32, order="C"))
    except Exception as exc:
        raise RuntimeError(
            "cuML KMeans failed while moving data to CUDA. Check that WSL can see the NVIDIA "
            "driver/GPU and that the installed driver supports the RAPIDS CUDA runtime."
        ) from exc
    return np.asarray(labels).astype(np.int64, copy=False)


def cuml_agglomerative_predict(
    values: np.ndarray,
    n_clusters: int,
    linkage: str = "single",
) -> np.ndarray:
    try:
        from cuml.cluster import AgglomerativeClustering as CuMLAgglomerativeClustering
    except ImportError as exc:
        raise RuntimeError(
            "cuml_agglomerative requested but RAPIDS cuML is not installed in this Python environment. "
            "Activate the Voxtral uv env and install `cuml-cu12`."
        ) from exc

    assert_cuml_cuda_available()
    if linkage != "single":
        raise ValueError("cuml_agglomerative currently supports only linkage='single'.")
    model = CuMLAgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        output_type="numpy",
    )
    labels = model.fit_predict(np.asarray(values, dtype=np.float32, order="C"))
    return np.asarray(labels).astype(np.int64, copy=False)


def cuml_dbscan_predict(
    values: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    try:
        from cuml.cluster import DBSCAN as CuMLDBSCAN
    except ImportError as exc:
        raise RuntimeError(
            "cuml_dbscan requested but RAPIDS cuML is not installed in this Python environment. "
            "Activate the Voxtral uv env and install `cuml-cu12`."
        ) from exc

    assert_cuml_cuda_available()
    model = CuMLDBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        output_type="numpy",
    )
    labels = model.fit_predict(np.asarray(values, dtype=np.float32, order="C"))
    return np.asarray(labels).astype(np.int64, copy=False)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    contingency = pd.crosstab(pd.Series(y_pred, name="cluster"), pd.Series(y_true, name="target"))
    if contingency.empty:
        return float("nan")
    return float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())


def safe_silhouette(
    values: np.ndarray,
    clusters: np.ndarray,
    sample_size: int | None = None,
    seed: int = 751,
) -> float:
    unique_count = len(np.unique(clusters))
    if unique_count < 2 or unique_count >= values.shape[0]:
        return float("nan")
    effective_sample_size = None
    if sample_size is not None and sample_size > 0 and sample_size < values.shape[0]:
        effective_sample_size = int(sample_size)
    return float(
        silhouette_score(
            values,
            clusters,
            sample_size=effective_sample_size,
            random_state=seed,
        )
    )


def supervised_metrics(
    values: np.ndarray,
    clusters: np.ndarray,
    target: np.ndarray,
    silhouette_sample_size: int | None = None,
    seed: int = 751,
) -> dict[str, float]:
    return {
        "silhouette": safe_silhouette(values, clusters, silhouette_sample_size, seed),
        "nmi": float(normalized_mutual_info_score(target, clusters)),
        "ari": float(adjusted_rand_score(target, clusters)),
        "purity": purity_score(target, clusters),
    }


def cluster_label_summary(
    labels: pd.DataFrame,
    assignment_column: str,
    label_column: str,
) -> pd.DataFrame:
    rows = []
    grouped = labels.groupby(assignment_column, sort=True)
    for cluster_id, group in grouped:
        counts = group[label_column].value_counts(dropna=False)
        top_label = counts.index[0] if not counts.empty else None
        top_count = int(counts.iloc[0]) if not counts.empty else 0
        rows.append(
            {
                "assignment": assignment_column,
                "label_column": label_column,
                "cluster": int(cluster_id),
                "size": int(len(group)),
                "top_label": top_label,
                "top_label_count": top_count,
                "top_label_fraction": float(top_count / len(group)) if len(group) else float("nan"),
                "unique_labels": int(group[label_column].nunique(dropna=False)),
            }
        )
    return pd.DataFrame(rows)
