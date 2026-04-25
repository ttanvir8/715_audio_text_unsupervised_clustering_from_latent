#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

RAW_DATASET_DIR="${PROJECT_ROOT}/0_data_pre_processing/raw_dataset"
RAW_DATASET_PATH="${RAW_DATASET_DIR}/4mula_small.parquet"
RAW_DATASET_URL="https://zenodo.org/records/4636802/files/4mula_small.parquet?download=1"
RAW_DATASET_MD5="30210cf6f52449c8d0670fc0942410c4"

WITH_GPU=0
WITH_OPTIONAL=0
DOWNLOAD_RAW=0
SKIP_VERIFY=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_env.sh [options]

Create a reproducible uv-managed Python environment for this project.

Options:
  --venv-dir PATH       Virtual environment path. Default: ./.venv
  --python VERSION      Python version for uv venv. Default: 3.12
  --with-gpu            Install RAPIDS/cuML CUDA 12 dependencies from requirements-gpu.txt
  --with-optional       Install optional CPU-side extras from requirements-optional.txt
  --download-raw        Download the 4MuLA raw parquet dataset into 0_data_pre_processing/raw_dataset/
  --skip-verify         Skip import verification after install
  -h, --help            Show this help

Examples:
  scripts/setup_env.sh
  scripts/setup_env.sh --with-gpu
  scripts/setup_env.sh --with-gpu --with-optional --download-raw
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --with-gpu)
      WITH_GPU=1
      shift
      ;;
    --with-optional)
      WITH_OPTIONAL=1
      shift
      ;;
    --download-raw)
      DOWNLOAD_RAW=1
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is not installed.

Install uv first, then rerun this script:
  curl -LsSf https://astral.sh/uv/install.sh | sh

After installation, open a new shell or source your shell profile so `uv` is on PATH.
EOF
  exit 1
fi

echo "[setup] Project root: ${PROJECT_ROOT}"
if [[ -d "${VENV_DIR}" ]]; then
  echo "[setup] Reusing existing venv: ${VENV_DIR}"
else
  echo "[setup] Creating uv venv: ${VENV_DIR}"
  uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
fi

ACTIVATE_PATH="${VENV_DIR}/bin/activate"
if [[ ! -f "${ACTIVATE_PATH}" ]]; then
  echo "Could not find venv activation script: ${ACTIVATE_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ACTIVATE_PATH}"

echo "[setup] Python: $(python --version)"
echo "[setup] Installing core requirements"
uv pip install -r "${PROJECT_ROOT}/requirements.txt"

if [[ "${WITH_OPTIONAL}" -eq 1 ]]; then
  echo "[setup] Installing optional CPU-side requirements"
  uv pip install -r "${PROJECT_ROOT}/requirements-optional.txt"
fi

if [[ "${WITH_GPU}" -eq 1 ]]; then
  echo "[setup] Installing optional GPU requirements"
  uv pip install -r "${PROJECT_ROOT}/requirements-gpu.txt"
fi

download_raw_dataset() {
  mkdir -p "${RAW_DATASET_DIR}"

  if [[ -f "${RAW_DATASET_PATH}" ]]; then
    echo "[setup] Raw dataset already exists: ${RAW_DATASET_PATH}"
  else
    echo "[setup] Downloading 4MuLA raw dataset. This file is about 12.8 GB."
    echo "[setup] Source: ${RAW_DATASET_URL}"
    if command -v curl >/dev/null 2>&1; then
      curl -L --fail --continue-at - --output "${RAW_DATASET_PATH}" "${RAW_DATASET_URL}"
    elif command -v wget >/dev/null 2>&1; then
      wget --continue --output-document="${RAW_DATASET_PATH}" "${RAW_DATASET_URL}"
    else
      echo "Neither curl nor wget is installed. Install one of them and rerun with --download-raw." >&2
      exit 1
    fi
  fi

  if command -v md5sum >/dev/null 2>&1; then
    echo "[setup] Verifying raw dataset MD5"
    actual_md5="$(md5sum "${RAW_DATASET_PATH}" | awk '{print $1}')"
    if [[ "${actual_md5}" != "${RAW_DATASET_MD5}" ]]; then
      echo "MD5 mismatch for ${RAW_DATASET_PATH}" >&2
      echo "Expected: ${RAW_DATASET_MD5}" >&2
      echo "Actual:   ${actual_md5}" >&2
      exit 1
    fi
    echo "[setup] Raw dataset MD5 ok"
  else
    echo "[setup] md5sum not found; skipping raw dataset checksum verification"
  fi
}

if [[ "${DOWNLOAD_RAW}" -eq 1 ]]; then
  download_raw_dataset
else
  mkdir -p "${RAW_DATASET_DIR}"
  echo "[setup] Raw dataset download skipped. Use --download-raw to fetch 4mula_small.parquet."
fi

if [[ "${SKIP_VERIFY}" -eq 0 ]]; then
  echo "[setup] Verifying core imports"
  python - <<'PY'
import duckdb
import librosa
import matplotlib
import numpy
import pandas
import polars
import pyarrow
import sklearn
import torch
import transformers
import yaml

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("core imports ok")
PY

  if [[ "${WITH_GPU}" -eq 1 ]]; then
    echo "[setup] Verifying cuML import"
    python - <<'PY'
import cuml
print("cuml:", cuml.__version__)
PY
  fi
fi

cat <<EOF

[setup] Done.

Activate this environment with:
  source "${ACTIVATE_PATH}"

Raw dataset target:
  ${RAW_DATASET_PATH}
EOF
