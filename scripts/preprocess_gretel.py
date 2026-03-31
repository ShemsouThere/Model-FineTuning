from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from text2sql_unsloth.config import ensure_dir, load_config
from text2sql_unsloth.prompting import build_messages
from text2sql_unsloth.sql_filters import (
    build_dedupe_key,
    canonicalize_schema_ddl,
    canonicalize_sqlite_query,
    extract_schema_ddl,
    has_blocklisted_dialect,
    in_spider_blocklist,
    is_read_only_sql,
    load_spider_blocklist,
    normalize_whitespace,
    parse_sqlite,
)


REQUIRED_COLUMNS = {
    "domain",
    "domain_description",
    "sql_complexity",
    "sql_task_type",
    "sql_prompt",
    "sql_context",
    "sql",
    "sql_explanation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Gretel synthetic Text-to-SQL data.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default=None, help="Optional override config.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--spider-blocklist-json", default=None)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def describe_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    task_counts = Counter(record["sql_task_type"] for record in records)
    complexity_counts = Counter(record["sql_complexity"] for record in records)
    return {
        "row_count": len(records),
        "task_type_counts": dict(task_counts.most_common()),
        "sql_complexity_counts": dict(complexity_counts.most_common()),
    }


def clean_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    spider_blocklist: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dataset_cfg = config["dataset"]
    include_explanation = bool(dataset_cfg.get("include_explanation", False))
    stats = Counter()
    cleaned: list[dict[str, Any]] = []

    for raw in records:
        stats["seen"] += 1

        prompt = normalize_whitespace(str(raw.get("sql_prompt", "")))
        sql = normalize_whitespace(str(raw.get("sql", "")))
        sql_context = str(raw.get("sql_context", "") or "")
        task_type = str(raw.get("sql_task_type", "") or "")

        if len(prompt) < dataset_cfg["min_prompt_chars"]:
            stats["drop_short_prompt"] += 1
            continue
        if len(sql) < dataset_cfg["min_sql_chars"]:
            stats["drop_short_sql"] += 1
            continue
        if task_type not in set(dataset_cfg["allowed_task_types"]):
            stats["drop_task_type"] += 1
            continue
        if not is_read_only_sql(
            sql,
            allowed_leading_keywords=dataset_cfg["allowed_leading_keywords"],
            blocked_keywords=dataset_cfg["blocked_sql_keywords"],
        ):
            stats["drop_non_read_only_sql"] += 1
            continue
        if has_blocklisted_dialect(sql, dataset_cfg["blocked_dialect_regex"]):
            stats["drop_blocklisted_dialect"] += 1
            continue
        if not parse_sqlite(sql):
            stats["drop_sqlite_parse_failed"] += 1
            continue

        schema_ddl = extract_schema_ddl(
            sql_context,
            keep_create_view=bool(dataset_cfg.get("keep_create_view", False)),
        )
        if dataset_cfg.get("require_schema_context", True) and not schema_ddl:
            stats["drop_missing_schema_ddl"] += 1
            continue

        canonical_sql = canonicalize_sqlite_query(sql)
        canonical_schema = canonicalize_schema_ddl(schema_ddl) if schema_ddl else ""

        if spider_blocklist and in_spider_blocklist(prompt, canonical_sql, spider_blocklist):
            stats["drop_spider_blocklist"] += 1
            continue

        item = {
            "domain": raw.get("domain"),
            "domain_description": raw.get("domain_description"),
            "sql_complexity": raw.get("sql_complexity"),
            "sql_task_type": task_type,
            "sql_prompt": prompt,
            "sql_context_raw": sql_context.strip(),
            "schema_ddl": canonical_schema,
            "sql": canonical_sql,
            "sql_explanation": normalize_whitespace(str(raw.get("sql_explanation", ""))),
        }
        item["messages"] = build_messages(
            question=item["sql_prompt"],
            schema=item["schema_ddl"],
            sql=item["sql"],
            sql_explanation=item["sql_explanation"],
            include_explanation=include_explanation,
            config=config,
            format_name=config["prompt"]["train_format"],
        )
        cleaned.append(item)
        stats["kept_before_dedupe"] += 1

    dedupe_fields = dataset_cfg["dedupe_on"]
    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in cleaned:
        dedupe_key = build_dedupe_key(item, dedupe_fields)
        if dedupe_key in seen_keys:
            stats["drop_duplicate"] += 1
            continue
        seen_keys.add(dedupe_key)
        deduped.append(item)

    stats["kept_after_dedupe"] = len(deduped)
    return deduped, dict(stats)


def split_records(records: list[dict[str, Any]], config: dict[str, Any], seed: int) -> dict[str, list[dict[str, Any]]]:
    split_cfg = config["dataset"]["split"]
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)

    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["val_ratio"])
    test_ratio = float(split_cfg["test_ratio"])
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config, args.config)
    if args.sample_size is not None:
        config["dataset"]["sample_size"] = args.sample_size
    if args.output_root is not None:
        config["dataset"]["output_root"] = args.output_root
    if args.seed is not None:
        config["seed"] = args.seed

    dataset_cfg = config["dataset"]
    seed = int(config["seed"])

    dataset = load_dataset(dataset_cfg["hf_name"], split=dataset_cfg["hf_split"])
    if not REQUIRED_COLUMNS.issubset(set(dataset.column_names)):
        missing = REQUIRED_COLUMNS - set(dataset.column_names)
        raise ValueError(f"Dataset schema changed. Missing columns: {sorted(missing)}")

    records = list(dataset)
    summary_before = describe_dataset(records)

    if args.inspect_only:
        print(json.dumps({
            "columns": dataset.column_names,
            "summary_before_filtering": summary_before,
            "example": {key: records[0][key] for key in sorted(REQUIRED_COLUMNS)},
        }, indent=2, ensure_ascii=False))
        return

    spider_blocklist = load_spider_blocklist(args.spider_blocklist_json)
    cleaned, filter_stats = clean_records(records, config, spider_blocklist)

    sample_size = dataset_cfg.get("sample_size")
    if sample_size:
        random.Random(seed).shuffle(cleaned)
        cleaned = cleaned[: int(sample_size)]

    split_rows = split_records(cleaned, config, seed)
    output_root = ensure_dir(dataset_cfg["output_root"])

    write_jsonl(output_root / "all_cleaned.jsonl", cleaned)
    for split_name, rows in split_rows.items():
        write_jsonl(output_root / f"{split_name}.jsonl", rows)

    summary = {
        "dataset_name": dataset_cfg["hf_name"],
        "split": dataset_cfg["hf_split"],
        "columns": dataset.column_names,
        "summary_before_filtering": summary_before,
        "filter_stats": filter_stats,
        "sample_size": len(cleaned),
        "split_sizes": {key: len(value) for key, value in split_rows.items()},
        "seed": seed,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

