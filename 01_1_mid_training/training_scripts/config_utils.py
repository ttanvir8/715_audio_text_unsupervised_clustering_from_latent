from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_experiment_config(experiment_config_path: str | Path) -> dict[str, Any]:
    experiment_path = project_path(experiment_config_path)
    experiment = load_yaml(experiment_path)

    data_config = load_yaml(project_path(experiment["data_config"]))
    vae_config = load_yaml(project_path(experiment["vae_config"]))

    return {
        "experiment": experiment,
        "data": data_config,
        "vae": vae_config,
        "config_paths": {
            "experiment_config": str(experiment_path),
            "data_config": str(project_path(experiment["data_config"])),
            "vae_config": str(project_path(experiment["vae_config"])),
        },
    }


def resolve_output_dir(config: dict[str, Any]) -> Path:
    return project_path(config["experiment"]["output_dir"])

