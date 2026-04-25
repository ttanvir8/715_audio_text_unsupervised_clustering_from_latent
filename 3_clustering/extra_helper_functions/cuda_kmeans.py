from __future__ import annotations

import math

import numpy as np
import torch


def _resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "cuda_kmeans requested a CUDA device, but torch.cuda.is_available() is false."
        )
    return torch.device(device_name)


def _squared_distances(
    values: torch.Tensor,
    centers: torch.Tensor,
    batch_size: int | None,
) -> torch.Tensor:
    if batch_size is None or batch_size <= 0 or batch_size >= values.shape[0]:
        return torch.cdist(values, centers, p=2).pow_(2)

    chunks = []
    for start in range(0, values.shape[0], batch_size):
        chunk = values[start : start + batch_size]
        chunks.append(torch.cdist(chunk, centers, p=2).pow_(2))
    return torch.cat(chunks, dim=0)


def _kmeans_plus_plus(
    values: torch.Tensor,
    n_clusters: int,
    generator: torch.Generator,
    batch_size: int | None,
) -> torch.Tensor:
    n_samples = values.shape[0]
    first_index = torch.randint(n_samples, (1,), device=values.device, generator=generator)
    centers = [values[first_index].squeeze(0)]
    closest_distances = _squared_distances(values, centers[0].unsqueeze(0), batch_size).squeeze(1)

    for _ in range(1, n_clusters):
        total_distance = closest_distances.sum()
        if not torch.isfinite(total_distance) or total_distance <= 0:
            next_index = torch.randint(n_samples, (1,), device=values.device, generator=generator)
        else:
            probabilities = closest_distances / total_distance
            next_index = torch.multinomial(probabilities, 1, generator=generator)

        center = values[next_index].squeeze(0)
        centers.append(center)
        candidate_distances = _squared_distances(values, center.unsqueeze(0), batch_size).squeeze(1)
        closest_distances = torch.minimum(closest_distances, candidate_distances)

    return torch.stack(centers, dim=0)


def _labels_and_inertia(
    values: torch.Tensor,
    centers: torch.Tensor,
    batch_size: int | None,
) -> tuple[torch.Tensor, float]:
    distances = _squared_distances(values, centers, batch_size)
    min_distances, labels = distances.min(dim=1)
    return labels, float(min_distances.sum().detach().cpu())


def _single_kmeans_run(
    values: torch.Tensor,
    n_clusters: int,
    generator: torch.Generator,
    max_iter: int,
    tol: float,
    batch_size: int | None,
) -> tuple[torch.Tensor, float]:
    centers = _kmeans_plus_plus(values, n_clusters, generator, batch_size)
    previous_inertia = math.inf

    for _ in range(max_iter):
        labels, inertia = _labels_and_inertia(values, centers, batch_size)
        new_centers = centers.clone()

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if bool(mask.any()):
                new_centers[cluster_id] = values[mask].mean(dim=0)
            else:
                replacement = torch.randint(
                    values.shape[0],
                    (1,),
                    device=values.device,
                    generator=generator,
                )
                new_centers[cluster_id] = values[replacement].squeeze(0)

        center_shift = torch.linalg.vector_norm(new_centers - centers).item()
        relative_improvement = abs(previous_inertia - inertia) / max(abs(previous_inertia), 1.0)
        centers = new_centers
        previous_inertia = inertia
        if center_shift <= tol or relative_improvement <= tol:
            break

    labels, inertia = _labels_and_inertia(values, centers, batch_size)
    return labels, inertia


def torch_kmeans_predict(
    values: np.ndarray,
    n_clusters: int,
    seed: int = 751,
    device: str = "cuda",
    n_init: int = 20,
    max_iter: int = 100,
    tol: float = 1e-4,
    batch_size: int | None = None,
) -> np.ndarray:
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if n_clusters > values.shape[0]:
        raise ValueError(f"n_clusters={n_clusters} exceeds sample count={values.shape[0]}.")

    torch_device = _resolve_device(device)
    tensor_values = torch.as_tensor(values, dtype=torch.float32, device=torch_device)
    best_labels = None
    best_inertia = math.inf

    for init_index in range(max(1, int(n_init))):
        generator = torch.Generator(device=torch_device)
        generator.manual_seed(int(seed) + init_index)
        labels, inertia = _single_kmeans_run(
            values=tensor_values,
            n_clusters=n_clusters,
            generator=generator,
            max_iter=max_iter,
            tol=tol,
            batch_size=batch_size,
        )
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.detach().cpu()

    if best_labels is None:
        raise RuntimeError("cuda_kmeans did not produce labels.")
    return best_labels.numpy().astype(np.int64, copy=False)
