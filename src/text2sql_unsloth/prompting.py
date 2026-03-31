from __future__ import annotations

from typing import Any


def build_user_message(question: str, schema: str, user_template: str) -> str:
    return user_template.format(
        question=question.strip(),
        schema=schema.strip(),
    ).strip()


def build_messages(
    *,
    question: str,
    schema: str,
    sql: str,
    config: dict[str, Any],
    include_explanation: bool = False,
    sql_explanation: str | None = None,
    format_name: str = "direct_sql",
) -> list[dict[str, str]]:
    prompt_cfg = config["prompt"]
    if format_name == "advanced_reasoning":
        system_prompt = prompt_cfg["advanced_reasoning_system_prompt"].strip()
    else:
        system_prompt = prompt_cfg["direct_sql_system_prompt"].strip()

    user_content = build_user_message(
        question=question,
        schema=schema,
        user_template=prompt_cfg["user_template"],
    )
    assistant_content = sql.strip()
    if include_explanation and sql_explanation:
        assistant_content = (
            "Reasoning:\n"
            f"{sql_explanation.strip()}\n\n"
            "Final SQL:\n"
            f"{sql.strip()}"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def render_chat(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def extract_sql_from_response(text: str) -> str:
    stripped = text.strip()
    if "```sql" in stripped:
        after = stripped.split("```sql", 1)[1]
        return after.split("```", 1)[0].strip()
    if "```" in stripped:
        after = stripped.split("```", 1)[1]
        return after.split("```", 1)[0].strip()

    final_sql_prefix = "Final SQL:"
    if final_sql_prefix in stripped:
        return stripped.split(final_sql_prefix, 1)[1].strip()
    return stripped

