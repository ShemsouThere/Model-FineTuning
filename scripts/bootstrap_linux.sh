#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SPIDER_ARCHIVE="${SPIDER_ARCHIVE:-}"
SPIDER_GDRIVE_URL="${SPIDER_GDRIVE_URL:-}"
SPIDER_DOWNLOAD_PATH="${SPIDER_DOWNLOAD_PATH:-$REPO_ROOT/data/raw/spider.zip}"
SPIDER_EXTRACT_ROOT="${SPIDER_EXTRACT_ROOT:-$REPO_ROOT/data/raw}"
INSTALL_LLAMA_CPP="${INSTALL_LLAMA_CPP:-0}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$REPO_ROOT/llama.cpp}"

echo "Repo root: $REPO_ROOT"
echo "Venv dir:  $VENV_DIR"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements/vast.txt"
python -m pip install -e "$REPO_ROOT"

download_spider_archive() {
  if [[ -n "$SPIDER_ARCHIVE" ]]; then
    return
  fi
  if [[ -z "$SPIDER_GDRIVE_URL" ]]; then
    return
  fi

  mkdir -p "$(dirname "$SPIDER_DOWNLOAD_PATH")"
  echo "Downloading Spider archive from Google Drive to $SPIDER_DOWNLOAD_PATH"
  python -m gdown --fuzzy "$SPIDER_GDRIVE_URL" -O "$SPIDER_DOWNLOAD_PATH"
  SPIDER_ARCHIVE="$SPIDER_DOWNLOAD_PATH"
}

download_spider_archive

if [[ -n "$SPIDER_ARCHIVE" ]]; then
  python "$REPO_ROOT/scripts/preprocess_spider.py" \
    --config "$REPO_ROOT/configs/spider1_qwen25_7b.yaml" \
    --spider-archive "$SPIDER_ARCHIVE" \
    --inspect-only
fi

if [[ "$INSTALL_LLAMA_CPP" == "1" ]]; then
  if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
  fi
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DGGML_CUDA=OFF
  cmake --build "$LLAMA_CPP_DIR/build" --config Release -j
fi

echo "Bootstrap complete."
