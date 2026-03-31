from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from text2sql_unsloth.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained adapter to merged or GGUF artifacts.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--gguf-dir", default=None)
    parser.add_argument("--gguf-quant", default=None)
    parser.add_argument("--llama-cpp-dir", default=None)
    return parser.parse_args()


def resolve_quantization(quant_name: str | None) -> str:
    if not quant_name:
        return "Q4_K_M"
    return quant_name.upper()


def find_llama_quantize_binary(llama_cpp_dir: Path) -> Path:
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize",
        llama_cpp_dir / "llama-quantize",
        llama_cpp_dir / "quantize",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find llama-quantize under {llama_cpp_dir}")


def export_merged_model(base_model: str, adapter_dir: str, merged_dir: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_path = Path(merged_dir)
    merged_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    return merged_path


def export_gguf(merged_dir: Path, gguf_dir: Path, quant_name: str, llama_cpp_dir: Path) -> Path:
    gguf_dir.mkdir(parents=True, exist_ok=True)
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Could not find convert_hf_to_gguf.py under {llama_cpp_dir}")

    f16_path = gguf_dir / "model-f16.gguf"
    subprocess.run(
        [
            "python",
            str(convert_script),
            str(merged_dir),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ],
        check=True,
    )

    quant_binary = find_llama_quantize_binary(llama_cpp_dir)
    quantized_path = gguf_dir / f"model-{quant_name.lower()}.gguf"
    subprocess.run(
        [
            str(quant_binary),
            str(f16_path),
            str(quantized_path),
            resolve_quantization(quant_name),
        ],
        check=True,
    )
    return quantized_path


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config, args.config)

    merged_path: Path | None = None

    if args.merged_dir:
        merged_path = export_merged_model(args.base_model, args.adapter_dir, args.merged_dir)
        print(f"Merged model saved to {merged_path}")

    if args.gguf_dir:
        if not merged_path:
            if not args.merged_dir:
                raise ValueError("--merged-dir is required when exporting GGUF.")
            merged_path = Path(args.merged_dir)
        llama_cpp_dir = Path(args.llama_cpp_dir) if args.llama_cpp_dir else Path("llama.cpp")
        gguf_path = export_gguf(
            merged_path,
            Path(args.gguf_dir),
            args.gguf_quant or config["export"]["gguf_quantization_method"],
            llama_cpp_dir,
        )
        print(f"GGUF saved to {gguf_path}")


if __name__ == "__main__":
    main()
