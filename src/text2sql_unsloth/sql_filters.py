from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import sqlglot


WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def normalize_text_key(text: str) -> str:
    return normalize_whitespace(text).lower()


def normalize_sql_for_dedupe(sql: str) -> str:
    sql = normalize_whitespace(sql)
    return sql.rstrip(";").lower()


def split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False

    for char in sql_text:
        current.append(char)
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def extract_schema_ddl(sql_context: str, *, keep_create_view: bool = False) -> str:
    statements = []
    for statement in split_sql_statements(sql_context):
        stripped = statement.strip().rstrip(";")
        upper = stripped.upper()
        if upper.startswith("CREATE TABLE"):
            statements.append(f"{stripped};")
        elif keep_create_view and upper.startswith("CREATE VIEW"):
            statements.append(f"{stripped};")
    return "\n".join(statements).strip()


def leading_keyword(sql: str) -> str:
    normalized = normalize_whitespace(sql).lstrip("(")
    match = re.match(r"([A-Za-z]+)", normalized)
    return (match.group(1) if match else "").upper()


def is_read_only_sql(sql: str, allowed_leading_keywords: list[str], blocked_keywords: list[str]) -> bool:
    upper = sql.upper()
    if leading_keyword(sql) not in set(allowed_leading_keywords):
        return False
    return not any(re.search(rf"\b{re.escape(keyword)}\b", upper) for keyword in blocked_keywords)


def has_blocklisted_dialect(sql: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, sql, flags=re.IGNORECASE) for pattern in patterns)


def parse_sqlite(sql: str) -> bool:
    try:
        sqlglot.parse_one(sql, read="sqlite")
        return True
    except Exception:
        return False


def canonicalize_sqlite_query(sql: str) -> str:
    expression = sqlglot.parse_one(sql, read="sqlite")
    return expression.sql(dialect="sqlite", pretty=False)


def canonicalize_schema_ddl(schema_ddl: str) -> str:
    canonicalized: list[str] = []
    for statement in split_sql_statements(schema_ddl):
        stripped = statement.strip().rstrip(";")
        if not stripped:
            continue
        try:
            expression = sqlglot.parse_one(stripped, read="sqlite")
            canonicalized.append(expression.sql(dialect="sqlite", pretty=False) + ";")
        except Exception:
            canonicalized.append(normalize_whitespace(stripped) + ";")
    return "\n".join(canonicalized).strip()


def load_spider_blocklist(path: str | Path | None) -> dict[str, set[str]]:
    if not path:
        return {"questions": set(), "sql": set(), "pairs": set()}

    data_path = Path(path)
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    questions: set[str] = set()
    sql_entries: set[str] = set()
    pairs: set[str] = set()

    for item in raw:
        question = normalize_text_key(str(item.get("question") or item.get("prompt") or ""))
        sql_value = normalize_sql_for_dedupe(str(item.get("sql") or item.get("query") or ""))
        if question:
            questions.add(question)
        if sql_value:
            sql_entries.add(sql_value)
        if question and sql_value:
            pairs.add(f"{question}|||{sql_value}")
    return {"questions": questions, "sql": sql_entries, "pairs": pairs}


def in_spider_blocklist(question: str, sql: str, blocklist: dict[str, set[str]]) -> bool:
    normalized_question = normalize_text_key(question)
    normalized_sql = normalize_sql_for_dedupe(sql)
    return (
        normalized_question in blocklist["questions"]
        or normalized_sql in blocklist["sql"]
        or f"{normalized_question}|||{normalized_sql}" in blocklist["pairs"]
    )


def build_dedupe_key(item: dict[str, Any], fields: list[str]) -> str:
    pieces = []
    for field in fields:
        value = item.get(field, "")
        if field == "sql":
            pieces.append(normalize_sql_for_dedupe(str(value)))
        else:
            pieces.append(normalize_text_key(str(value)))
    return "|||".join(pieces)
