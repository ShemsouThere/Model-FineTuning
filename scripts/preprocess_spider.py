from __future__ import annotations

import argparse
import json
import random
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from text2sql_unsloth.config import ensure_dir, load_config
from text2sql_unsloth.prompting import build_messages
from text2sql_unsloth.sql_filters import canonicalize_sqlite_query, normalize_whitespace, parse_sqlite


SPIDER_TYPE_MAP = {
    "text": "TEXT",
    "number": "REAL",
    "time": "TEXT",
    "boolean": "INTEGER",
    "others": "TEXT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Spider 1.0 into the chat-training format used by this repo.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--spider-root", default=None, help="Directory containing tables.json and train_spider.json.")
    parser.add_argument("--spider-archive", default=None, help="Optional .zip/.tar(.gz) archive to extract before preprocessing.")
    parser.add_argument("--tables-file", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def unpack_archive(archive_path: Path, extract_root: Path) -> Path:
    ensure_dir(extract_root)
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_root)
    elif archive_path.suffix.lower() in {".gz", ".tgz", ".tar"} or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(extract_root)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    return extract_root


def find_spider_root(root: Path) -> Path:
    if (root / "tables.json").exists() and (root / "train_spider.json").exists():
        return root

    candidates = []
    for tables_file in root.rglob("tables.json"):
        candidate = tables_file.parent
        if (candidate / "train_spider.json").exists():
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"Could not locate Spider files under {root}")
    return sorted(candidates)[0]


def sqlite_type(spider_type: str) -> str:
    return SPIDER_TYPE_MAP.get((spider_type or "").lower(), "TEXT")


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def render_schema_ddl(schema: dict[str, Any]) -> str:
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]
    column_types = schema["column_types"]

    pk_by_table: dict[int, list[int]] = {}
    for column_idx in schema.get("primary_keys", []):
        table_idx = column_names[column_idx][0]
        pk_by_table.setdefault(table_idx, []).append(column_idx)

    fk_by_table: dict[int, list[tuple[int, int]]] = {}
    for source_idx, target_idx in schema.get("foreign_keys", []):
        table_idx = column_names[source_idx][0]
        fk_by_table.setdefault(table_idx, []).append((source_idx, target_idx))

    statements: list[str] = []
    for table_idx, table_name in enumerate(table_names):
        lines: list[str] = []
        table_constraints: list[str] = []
        table_pks = pk_by_table.get(table_idx, [])

        for column_idx, (column_table_idx, column_name) in enumerate(column_names):
            if column_table_idx != table_idx:
                continue
            column_def = f"  {quote_ident(column_name)} {sqlite_type(column_types[column_idx])}"
            if len(table_pks) == 1 and column_idx in table_pks:
                column_def += " PRIMARY KEY"
            lines.append(column_def)

        if len(table_pks) > 1:
            pk_columns = ", ".join(quote_ident(column_names[column_idx][1]) for column_idx in table_pks)
            table_constraints.append(f"  PRIMARY KEY ({pk_columns})")

        for source_idx, target_idx in fk_by_table.get(table_idx, []):
            target_table_idx, target_column_name = column_names[target_idx]
            source_column_name = column_names[source_idx][1]
            target_table_name = table_names[target_table_idx]
            table_constraints.append(
                "  FOREIGN KEY ({source}) REFERENCES {target_table} ({target})".format(
                    source=quote_ident(source_column_name),
                    target_table=quote_ident(target_table_name),
                    target=quote_ident(target_column_name),
                )
            )

        body = ",\n".join(lines + table_constraints)
        statements.append(f"CREATE TABLE {quote_ident(table_name)} (\n{body}\n);")

    return "\n".join(statements)


def load_spider_tables(tables_file: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(tables_file.read_text(encoding="utf-8"))
    return {item["db_id"]: item for item in raw}


def describe_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    db_counts = Counter(record["db_id"] for record in records)
    sql_parse_failures = sum(0 if parse_sqlite(record["query"]) else 1 for record in records)
    return {
        "row_count": len(records),
        "database_count": len(db_counts),
        "top_databases": dict(db_counts.most_common(10)),
        "sqlite_parse_failures": sql_parse_failures,
    }


def prepare_records(records: list[dict[str, Any]], tables_by_db: dict[str, dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    include_explanation = bool(config["dataset"].get("include_explanation", False))
    stats = Counter()
    prepared: list[dict[str, Any]] = []

    for raw in records:
        stats["seen"] += 1
        question = normalize_whitespace(str(raw.get("question", "")))
        sql = normalize_whitespace(str(raw.get("query", "")))
        db_id = str(raw.get("db_id", ""))
        if not question or not sql or not db_id:
            stats["drop_missing_fields"] += 1
            continue

        schema = tables_by_db.get(db_id)
        if not schema:
            stats["drop_missing_schema"] += 1
            continue

        schema_ddl = render_schema_ddl(schema)
        if parse_sqlite(sql):
            canonical_sql = canonicalize_sqlite_query(sql)
        else:
            canonical_sql = sql.rstrip(";")
            stats["sqlite_parse_failed"] += 1

        item = {
            "db_id": db_id,
            "sql_prompt": question,
            "schema_ddl": schema_ddl,
            "sql": canonical_sql,
            "messages": build_messages(
                question=question,
                schema=schema_ddl,
                sql=canonical_sql,
                sql_explanation=None,
                include_explanation=include_explanation,
                config=config,
                format_name=config["prompt"]["train_format"],
            ),
        }
        prepared.append(item)
        stats["kept"] += 1

    return prepared, dict(stats)


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
    if args.output_root:
        config["dataset"]["output_root"] = args.output_root
    if args.seed is not None:
        config["seed"] = args.seed

    raw_root = Path(config["dataset"].get("spider_root", "data/raw/spider1"))
    if args.spider_root:
        raw_root = Path(args.spider_root)

    if args.spider_archive:
        archive_path = Path(args.spider_archive)
        extract_root = raw_root.parent if raw_root.suffix else raw_root
        unpack_archive(archive_path, extract_root)
        raw_root = find_spider_root(extract_root)
    else:
        raw_root = find_spider_root(raw_root)

    tables_file = Path(args.tables_file) if args.tables_file else raw_root / "tables.json"
    train_file = Path(args.train_file) if args.train_file else raw_root / "train_spider.json"

    tables_by_db = load_spider_tables(tables_file)
    records = json.loads(train_file.read_text(encoding="utf-8"))
    summary_before = describe_records(records)

    if args.inspect_only:
        print(json.dumps({
            "spider_root": str(raw_root),
            "tables_file": str(tables_file),
            "train_file": str(train_file),
            "summary_before_processing": summary_before,
            "example": {
                "db_id": records[0]["db_id"],
                "question": records[0]["question"],
                "query": records[0]["query"],
            },
        }, indent=2, ensure_ascii=False))
        return

    prepared, prepare_stats = prepare_records(records, tables_by_db, config)
    split_rows = split_records(prepared, config, int(config["seed"]))
    output_root = ensure_dir(config["dataset"]["output_root"])

    write_jsonl(output_root / "all_cleaned.jsonl", prepared)
    for split_name, rows in split_rows.items():
        write_jsonl(output_root / f"{split_name}.jsonl", rows)

    summary = {
        "dataset_name": "spider1_official",
        "source_root": str(raw_root),
        "tables_file": str(tables_file),
        "train_file": str(train_file),
        "summary_before_processing": summary_before,
        "prepare_stats": prepare_stats,
        "sample_size": len(prepared),
        "split_sizes": {key: len(value) for key, value in split_rows.items()},
        "seed": int(config["seed"]),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
