#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO_ROOT/configs/spider1_qwen25_7b.yaml}"
BASE_CONFIG="${BASE_CONFIG:-$REPO_ROOT/configs/base.yaml}"
SPIDER_ROOT="${SPIDER_ROOT:-$REPO_ROOT/data/raw/spider1}"
SPIDER_ARCHIVE="${SPIDER_ARCHIVE:-}"
SPIDER_GDRIVE_URL="${SPIDER_GDRIVE_URL:-}"
SPIDER_DOWNLOAD_PATH="${SPIDER_DOWNLOAD_PATH:-$REPO_ROOT/data/raw/spider.zip}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_PACKAGE="${RUN_PACKAGE:-1}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"
GGUF_QUANT="${GGUF_QUANT:-q4_k_m}"
PYTHON_EXEC="${PYTHON_EXEC:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_EXEC" ]]; then
  PYTHON_EXEC="python"
fi

cd "$REPO_ROOT"

download_spider_archive() {
  if [[ -n "$SPIDER_ARCHIVE" ]]; then
    return
  fi
  if [[ -z "$SPIDER_GDRIVE_URL" ]]; then
    return
  fi

  mkdir -p "$(dirname "$SPIDER_DOWNLOAD_PATH")"
  echo "Downloading Spider archive from Google Drive to $SPIDER_DOWNLOAD_PATH"
  "$PYTHON_EXEC" -m gdown --fuzzy "$SPIDER_GDRIVE_URL" -O "$SPIDER_DOWNLOAD_PATH"
  SPIDER_ARCHIVE="$SPIDER_DOWNLOAD_PATH"
}

download_spider_archive

PREPROCESS_ARGS=(--base-config "$BASE_CONFIG" --config "$CONFIG" --spider-root "$SPIDER_ROOT")
if [[ -n "$SPIDER_ARCHIVE" ]]; then
  PREPROCESS_ARGS+=(--spider-archive "$SPIDER_ARCHIVE")
fi

"$PYTHON_EXEC" "$REPO_ROOT/scripts/preprocess_spider.py" "${PREPROCESS_ARGS[@]}"
"$PYTHON_EXEC" -u "$REPO_ROOT/scripts/train_unsloth.py" --base-config "$BASE_CONFIG" --config "$CONFIG"

if [[ "$RUN_EXPORT" == "1" ]]; then
  OUTPUT_DIR="$("$PYTHON_EXEC" - "$BASE_CONFIG" "$CONFIG" <<'PY'
import sys
from text2sql_unsloth.config import load_config
config = load_config(sys.argv[1], sys.argv[2])
print(config["training"]["output_dir"])
PY
)"

  EXPORT_ARGS=(
    --base-config "$BASE_CONFIG"
    --config "$CONFIG"
    --adapter-dir "$OUTPUT_DIR/adapter"
    --base-model "$BASE_MODEL"
    --merged-dir "$OUTPUT_DIR/merged_16bit"
  )

  if [[ -n "$LLAMA_CPP_DIR" ]]; then
    EXPORT_ARGS+=(--gguf-dir "$OUTPUT_DIR/gguf" --gguf-quant "$GGUF_QUANT" --llama-cpp-dir "$LLAMA_CPP_DIR")
  fi

  "$PYTHON_EXEC" "$REPO_ROOT/scripts/export_model.py" "${EXPORT_ARGS[@]}"

  if [[ "$RUN_PACKAGE" == "1" ]]; then
    "$PYTHON_EXEC" "$REPO_ROOT/scripts/package_artifacts.py" --source-dir "$OUTPUT_DIR" --format zip
  fi
fi

echo "Spider 1.0 Qwen2.5-Coder run complete."
