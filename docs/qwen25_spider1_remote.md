# Qwen2.5-Coder Spider 1.0 Remote Run

This runbook is for a one-shot rented GPU session where you want to:

1. bootstrap a fresh Linux box
2. preprocess Spider 1.0
3. fine-tune `Qwen2.5-Coder-7B-Instruct` with Unsloth
4. export a merged model
5. optionally export GGUF if `llama.cpp` is installed
6. package the full run directory so you can shut the server down

## Recommended GPU

For a 1 epoch Spider 1.0 QLoRA run with `Qwen2.5-Coder-7B-Instruct`:

- Best chance of finishing training in under 1 hour: `A100 40GB`, `A100 80GB`
- Usually workable in about 1 hour to 2 hours: `L40S`, `A40`, `RTX 4090`
- More likely to drift toward 2 hours or beyond: `A10`, `L4`, `T4`

Use a machine with:

- at least 24 GB VRAM
- at least 100 GB free disk
- stable network for model download

## Files you need

- this repository
- a Spider 1.0 archive or extracted directory containing `tables.json` and `train_spider.json`
- optional: a Google Drive link to `spider.zip`

## Fresh machine setup

```bash
git clone <your-repo-url>
cd Model-FineTuning

chmod +x scripts/bootstrap_linux.sh
chmod +x scripts/run_spider1_qwen25.sh

SPIDER_ARCHIVE=/workspace/spider.zip bash scripts/bootstrap_linux.sh
```

If your archive lives on Google Drive, the scripts can download it directly with `gdown`:

```bash
SPIDER_GDRIVE_URL="https://drive.google.com/file/d/FILE_ID/view?usp=sharing" bash scripts/bootstrap_linux.sh
```

If you also want GGUF generation on the same machine:

```bash
SPIDER_ARCHIVE=/workspace/spider.zip INSTALL_LLAMA_CPP=1 bash scripts/bootstrap_linux.sh
```

## Run the training job

```bash
bash scripts/run_spider1_qwen25.sh
```

If your Spider archive is not already extracted:

```bash
SPIDER_ARCHIVE=/workspace/spider.zip bash scripts/run_spider1_qwen25.sh
```

If you want the run script to download it itself:

```bash
SPIDER_GDRIVE_URL="https://drive.google.com/file/d/FILE_ID/view?usp=sharing" bash scripts/run_spider1_qwen25.sh
```

If you also built `llama.cpp` and want GGUF:

```bash
LLAMA_CPP_DIR=/workspace/Model-FineTuning/llama.cpp bash scripts/run_spider1_qwen25.sh
```

## Outputs

The run directory is:

```text
artifacts/spider1_qwen25_7b
```

Expected contents:

- `adapter/`
- `merged_16bit/`
- `checkpoints/`
- `run_config.json`
- `gguf/` if GGUF export was enabled

The packaging step also creates:

- `artifacts/spider1_qwen25_7b.zip`

Download that archive before shutting the machine down.
