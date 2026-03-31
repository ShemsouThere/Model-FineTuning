from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from text2sql_unsloth.config import load_config
from text2sql_unsloth.prompting import build_messages, extract_sql_from_response, render_chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with transformers + PEFT on a trained LoRA adapter.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
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


def load_model(base_model: str, adapter_dir: str) -> tuple[Any, Any]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config, args.config)
    schema = load_schema(args)
    strategy = args.strategy or config["prompt"]["inference_format"]

    model, tokenizer = load_model(args.base_model, args.adapter_dir)
    messages = build_messages(
        question=args.question,
        schema=schema,
        sql="",
        config=config,
        include_explanation=False,
        format_name=strategy,
    )[:-1]
    prompt_text = render_chat(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(extract_sql_from_response(decoded))


if __name__ == "__main__":
    main()
