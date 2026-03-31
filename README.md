# Text-to-SQL Fine-Tuning with Unsloth

This project builds a reproducible QLoRA fine-tuning pipeline for `Qwen/Qwen2.5-Coder-7B-Instruct` using Unsloth, starting with a 20-row Google Colab smoke test and scaling to a 5,000-row Vast.ai run.

The training target is Spider 1.0 style Text-to-SQL. The preprocessing path explicitly filters the Gretel synthetic dataset toward read-only, SQLite-compatible query generation and strips row-value inserts from the input context so the model learns `question + schema -> SQL`, which is closer to Spider.

## Project layout

```text
.
+-- configs/
|   +-- base.yaml
|   +-- colab_smoke.yaml
|   `-- vast_5k.yaml
+-- data/
|   +-- raw/
|   `-- processed/
+-- requirements/
|   +-- common.txt
|   +-- colab.txt
|   `-- vast.txt
+-- output/jupyter-notebook/
|   `-- colab-qwen25-text2sql-smoke.ipynb
+-- scripts/
|   +-- export_model.py
|   +-- infer_unsloth.py
|   +-- preprocess_gretel.py
|   `-- train_unsloth.py
`-- src/text2sql_unsloth/
    +-- config.py
    +-- prompting.py
    `-- sql_filters.py
```

## Colab notebooks

Available notebook entrypoints:

- `output/jupyter-notebook/colab-qwen25-text2sql-smoke.ipynb`
- `output/jupyter-notebook/colab-qwen25-text2sql-all-in-one.ipynb`

Recommended default:

- `colab-qwen25-text2sql-all-in-one.ipynb`

It can:

- start from `/content/Model-FineTuning.zip`, a repo clone, or a fresh empty Colab workspace
- rewrite the core project files to the current Colab-compatible versions
- switch between a 20-row smoke test and a 5k run by changing one parameter cell
- preprocess, train, run post-training inference, and package the adapter artifacts

The smoke notebook still exists as a smaller reference flow:

- bootstrap the repo inside Colab
- install dependencies
- inspect the Gretel dataset schema
- preprocess 20 cleaned rows
- fine-tune the Unsloth QLoRA adapter
- run a smoke-test inference query
- zip the adapter artifact

## Dataset assumptions and filtering

The HF dataset card for `gretelai/synthetic_text_to_sql` exposes these columns:

- `domain`
- `domain_description`
- `sql_complexity`
- `sql_complexity_description`
- `sql_task_type`
- `sql_task_type_description`
- `sql_prompt`
- `sql_context`
- `sql`
- `sql_explanation`

This pipeline uses:

- Input candidates: `sql_prompt` and schema-only content extracted from `sql_context`
- Target: `sql`
- Optional later-stage reasoning target: `sql_explanation`

Default filtering rules:

- Keep only `sql_task_type == analytics and reporting`
- Keep only targets whose leading keyword is `SELECT` or `WITH`
- Drop targets containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `MERGE`, `TRUNCATE`, `GRANT`, `REVOKE`
- Drop obvious non-SQLite / non-Spider patterns such as `DATE_SUB`, `CURDATE`, `ILIKE`, `QUALIFY`, `UNNEST`, `LISTAGG`, `STRING_AGG`, `TOP n`, `APPLY`, `INTERVAL`
- Require the target query to parse under `sqlglot` with `read="sqlite"`
- Strip `INSERT` rows from `sql_context` and keep only `CREATE TABLE` statements as `schema_ddl`
- Deduplicate on normalized `(sql_prompt, schema_ddl, sql)`

This is intentionally stricter than the original dataset. You want benchmark alignment more than raw sample count.

## Training format

The default first-pass training format is direct supervised SQL generation, not explanation generation:

```text
system: You are an expert Text-to-SQL assistant. Return SQL only.
user:
Database schema:
CREATE TABLE singer (...);
CREATE TABLE concert (...);

Question:
List singer names ordered by age descending.

Return only SQLite SQL.
assistant:
SELECT name FROM singer ORDER BY age DESC;
```

This is the recommended starting point for Spider. It keeps outputs deterministic and avoids contaminating benchmark inference with explanation text.

For your `advanced_reasoning` inference strategy, keep the training target as SQL-only first. Use the reasoning-heavy prompt at inference time, but still require `SQL only` output. Add `sql_explanation` only in a second experiment after you have a clean baseline.

## Environment setup

### Google Colab smoke test

Use a T4, L4, A100, or better. T4 is enough for the 20-row smoke run.

```bash
git clone <your-repo-url>
cd Model-FineTuning
pip install -U pip
pip install -r requirements/colab.txt
pip install -e .
```

If a Colab runtime has trouble with the packaged `unsloth`, use the current upstream recommendation:

```bash
pip uninstall -y unsloth unsloth_zoo
pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --upgrade --no-cache-dir "git+https://github.com/unslothai/unsloth-zoo.git"
```

### Vast.ai

For the real 5k run, prefer:

- Minimum: 24 GB VRAM (`RTX 4090`, `RTX 3090`, `L40`, `A5000`)
- Better: 48 GB VRAM (`A6000`, `L40S`, `A40`)
- Disk: 80 GB+
- CUDA: a modern PyTorch-compatible image

Setup:

```bash
git clone <your-repo-url>
cd Model-FineTuning
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements/vast.txt
pip install -e .
```

## Step 1: inspect the dataset schema

```bash
python scripts/preprocess_gretel.py --inspect-only
```

This prints current columns, raw task-type distribution, complexity distribution, and one example. Run this first any time the dataset version might have changed.

## Step 2: create the 20-row Colab smoke dataset

```bash
python scripts/preprocess_gretel.py \
  --config configs/colab_smoke.yaml
```

Outputs:

- `data/processed/colab_smoke/all_cleaned.jsonl`
- `data/processed/colab_smoke/train.jsonl`
- `data/processed/colab_smoke/val.jsonl`
- `data/processed/colab_smoke/summary.json`

Expected split: about 16 train / 4 val.

## Step 3: train the Colab smoke model

```bash
python scripts/train_unsloth.py \
  --config configs/colab_smoke.yaml
```

Recommended Colab smoke hyperparameters:

- LoRA rank: `16`
- LR: `2e-4`
- Epochs: `10`
- Batch size: `1`
- Gradient accumulation: `4`
- Max length: `1536`

This run is supposed to overfit slightly. Its purpose is to verify:

- prompt formatting
- tokenization
- assistant-only label masking
- checkpoint saving
- adapter saving
- inference path

## Step 4: smoke-test inference

```bash
python scripts/infer_unsloth.py \
  --config configs/colab_smoke.yaml \
  --model-path artifacts/colab_smoke_qwen25_coder/adapter \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --question "What is the average property size in inclusive housing areas?" \
  --schema-text "CREATE TABLE Inclusive_Housing (Property_ID INT, Inclusive TEXT, Property_Size INT);" \
  --strategy advanced_reasoning
```

You should see SQL only.

## Step 5: build the 5k Vast.ai dataset

```bash
python scripts/preprocess_gretel.py \
  --config configs/vast_5k.yaml
```

Expected split: about 4500 train / 250 val / 250 test.

If you want an additional exact-match leak guard against Spider dev, prepare a JSON array like:

```json
[
  {"question": "How many singers are there?", "sql": "SELECT COUNT(*) FROM singer"},
  {"question": "List stadium names", "sql": "SELECT name FROM stadium"}
]
```

Then run:

```bash
python scripts/preprocess_gretel.py \
  --config configs/vast_5k.yaml \
  --spider-blocklist-json spider_dev_blocklist.json
```

This is optional, but it is the cleanest way to guarantee you are not fine-tuning on exact Spider dev duplicates if you maintain a local canonicalized question/SQL index.

## Step 6: train the 5k Vast.ai run

```bash
python scripts/train_unsloth.py \
  --config configs/vast_5k.yaml
```

Recommended starting hyperparameters for 5k rows:

- LoRA rank: `32`
- LoRA alpha: `64`
- LR: `1e-4`
- Epochs: `1` to start
- Batch size: `2`
- Gradient accumulation: `8`
- Effective batch size: `16`
- Max length: `2048`
- Optimizer: `adamw_8bit`

### Should you start with 1 epoch or 3 epochs?

Start with `1 epoch`.

Why:

- 5k rows is still small enough to overfit a 7B instruct model with QLoRA
- the dataset is synthetic, so extra epochs can amplify synthetic artifacts
- you care about Spider 1.0 transfer, not in-domain training loss

Decision rule:

- Run `1 epoch` first
- Check Spider dev execution accuracy / exact set match with your benchmark
- Only move to `2-3 epochs` if dev performance improves without a clear increase in malformed or overfit outputs

## Saving, merging, exporting

During training the pipeline can save:

- LoRA adapter: `artifacts/.../adapter`
- merged 16-bit model: `artifacts/.../merged_16bit`
- GGUF: optional

The `vast_5k.yaml` preset enables merged 16-bit export automatically.

Manual export after training:

```bash
python scripts/export_model.py \
  --config configs/vast_5k.yaml \
  --adapter-dir artifacts/vast_5k_qwen25_coder/adapter \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --merged-dir artifacts/vast_5k_qwen25_coder/merged_16bit
```

Optional GGUF export:

```bash
python scripts/export_model.py \
  --config configs/vast_5k.yaml \
  --adapter-dir artifacts/vast_5k_qwen25_coder/adapter \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --gguf-dir artifacts/vast_5k_qwen25_coder/gguf \
  --gguf-quant q4_k_m
```

If Unsloth GGUF export is unstable in your environment, fallback:

1. Export `merged_16bit`
2. Convert with your existing `llama.cpp` GGUF conversion flow
3. Benchmark that GGUF in your current Spider pipeline

## Using the fine-tuned model with your Spider 1.0 benchmark

Given that your current benchmark stack already runs local GGUF + `llama.cpp`, the clean handoff is:

1. Fine-tune with this project and save the adapter.
2. Export a merged 16-bit model.
3. Convert the merged model to GGUF if your benchmark expects GGUF.
4. Keep your Spider evaluation prompt as close as possible to the training format:

```text
Database schema:
...

Question:
...

Return only SQLite SQL.
```

5. For the `advanced_reasoning` benchmark variant, change only the system instruction, not the final output contract.

Recommended benchmark comparison set:

- Base `Qwen2.5-Coder-7B-Instruct`
- Fine-tuned adapter / merged model with direct inference prompt
- Fine-tuned adapter / merged model with advanced_reasoning prompt

That gives you a clean ablation on whether the adapter helps, whether the reasoning prompt helps, and whether the two combine.

## Qwen2.5-Coder / Unsloth caveats

- Qwen2.5 uses a ChatML-style template with `<|im_start|>` / `<|im_end|>`. This project relies on the tokenizer chat template directly instead of hard-coding prompt tokens.
- The pipeline uses custom label masking so only assistant SQL tokens contribute to the loss. This avoids depending on helper utilities that have changed across Unsloth releases.
- Keep `return SQL only` in both training and inference prompts. If you later add explanation training, do it as a separate experiment.
- Do not train on Gretel rows where the target SQL uses MySQL-style date arithmetic or other non-SQLite features if Spider 1.0 is the target benchmark.

## Common failure points and debugging

### 1. OOM on Colab / Vast

Try in this order:

- reduce `max_seq_length` from `2048` to `1536`
- reduce `per_device_train_batch_size` from `2` to `1`
- increase `gradient_accumulation_steps`
- keep LoRA rank at `16` for the smoke test

### 2. Model outputs explanations instead of pure SQL

- verify training used `include_explanation: false`
- keep inference prompt strict: `Return only SQLite SQL`
- lower temperature to `0.0`

### 3. SQLite-incompatible generations on Spider

- tighten `blocked_dialect_regex`
- inspect malformed Gretel examples that passed preprocessing
- increase filtering strictness before increasing training size

### 4. No learning signal on the 20-row smoke run

- inspect `data/processed/colab_smoke/train.jsonl`
- confirm `messages` contain the expected schema/question/SQL triplets
- verify training loss decreases across the small run
- test with a sample seen during training before testing a fresh example

### 5. Adapter loads but inference is bad

- make sure you passed `--base-model Qwen/Qwen2.5-Coder-7B-Instruct`
- confirm the tokenizer from the same base model is being used
- compare with the base model on the exact same prompt

## Recommended next experiment after the baseline

Only after the direct-SQL baseline is stable:

1. Duplicate `configs/vast_5k.yaml`
2. Set `dataset.include_explanation: true`
3. Change the assistant target format to:
   `Reasoning: ...`
   `Final SQL: ...`
4. Keep benchmark post-processing strict so only the final SQL is scored

That is the right place to test whether explicit reasoning supervision helps your `advanced_reasoning` inference strategy.
