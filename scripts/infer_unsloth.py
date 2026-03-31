from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from text2sql_unsloth.config import load_config
from text2sql_unsloth.prompting import build_messages, extract_sql_from_response, render_chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Unsloth Text-to-SQL model.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-path", required=True, help="Adapter dir or merged model dir.")
    parser.add_argument("--base-model", default=None, help="Required when --model-path is an adapter directory.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--schema-file", default=None)
    parser.add_argument("--schema-text", default=None)
    parser.add_argument("--strategy", choices=["direct_sql", "advanced_reasoning"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def load_schema(args: argparse.Namespace) -> str:
    if args.schema_text:
        return args.schema_text.strip()
    if args.schema_file:
        return Path(args.schema_file).read_text(encoding="utf-8").strip()
    raise ValueError("Provide either --schema-file or --schema-text.")


def load_model(model_path: str, base_model: str | None, config: dict[str, Any]) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    model_cfg = config["model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config, args.config)
    schema = load_schema(args)
    strategy = args.strategy or config["prompt"]["inference_format"]

    model, tokenizer = load_model(args.model_path, args.base_model, config)

    messages = build_messages(
        question=args.question,
        schema=schema,
        sql="",
        config=config,
        include_explanation=False,
        format_name=strategy,
    )[:-1]
    prompt_text = render_chat(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            use_cache=True,
        )

    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    print(extract_sql_from_response(decoded))


if __name__ == "__main__":
    main()
