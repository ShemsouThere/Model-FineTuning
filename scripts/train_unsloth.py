from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import unsloth
import torch
from datasets import Dataset, load_dataset
from transformers import Trainer, TrainingArguments

from text2sql_unsloth.config import ensure_dir, load_config
from text2sql_unsloth.prompting import render_chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5-Coder-7B-Instruct with Unsloth QLoRA.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_splits(config: dict[str, Any], train_file: str | None, val_file: str | None) -> tuple[Dataset, Dataset]:
    dataset_root = Path(config["dataset"]["output_root"])
    train_path = train_file or str(dataset_root / "train.jsonl")
    val_path = val_file or str(dataset_root / "val.jsonl")
    data_files = {"train": train_path}
    if Path(val_path).exists():
        data_files["validation"] = val_path
    dataset_dict = load_dataset("json", data_files=data_files)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict.get("validation", dataset_dict["train"].select(range(min(8, len(dataset_dict["train"])))))
    return train_dataset, eval_dataset


def tokenize_example(example: dict[str, Any], tokenizer: Any, max_seq_length: int) -> dict[str, Any]:
    messages = example["messages"]
    prompt_messages = messages[:-1]

    full_text = render_chat(tokenizer, messages, add_generation_prompt=False)
    prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)

    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )
    prompt_tokens = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_length,
    )

    prompt_len = len(prompt_tokens["input_ids"])
    input_ids = full_tokens["input_ids"]
    labels = list(input_ids)

    if prompt_len >= len(labels):
        return {"skip_example": True}

    labels[:prompt_len] = [-100] * prompt_len
    return {
        "input_ids": input_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels,
        "skip_example": False,
    }


class SupervisedDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_length)
            attention_masks.append(feature["attention_mask"] + [0] * pad_length)
            labels.append(feature["labels"] + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def save_run_manifest(config: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.base_config, args.config)
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    model_cfg = config["model"]
    train_cfg = config["training"]
    output_dir = ensure_dir(train_cfg["output_dir"])
    save_run_manifest(config, output_dir)

    from unsloth import FastLanguageModel, is_bfloat16_supported

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model_name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),
        load_in_4bit=bool(model_cfg["load_in_4bit"]),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_cfg = model_cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["r"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias=str(lora_cfg["bias"]),
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=int(config["seed"]),
        max_seq_length=int(model_cfg["max_seq_length"]),
        use_rslora=bool(lora_cfg.get("use_rslora", False)),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_dataset, eval_dataset = load_splits(config, args.train_file, args.val_file)

    train_dataset = train_dataset.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": model_cfg["max_seq_length"]},
        remove_columns=train_dataset.column_names,
        num_proc=config["runtime"]["dataset_num_proc"],
    ).filter(lambda example: not example["skip_example"])
    eval_dataset = eval_dataset.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_length": model_cfg["max_seq_length"]},
        remove_columns=eval_dataset.column_names,
        num_proc=config["runtime"]["dataset_num_proc"],
    ).filter(lambda example: not example["skip_example"])

    print(f"Train rows after tokenization: {len(train_dataset)}")
    print(f"Eval rows after tokenization:  {len(eval_dataset)}")

    train_args_kwargs = dict(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=int(train_cfg["max_steps"]) if train_cfg["max_steps"] else -1,
        logging_steps=int(train_cfg["logging_steps"]),
        eval_steps=int(train_cfg["eval_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        save_total_limit=int(train_cfg["save_total_limit"]),
        save_strategy="steps",
        report_to=[] if train_cfg["report_to"] == "none" else [train_cfg["report_to"]],
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim=str(train_cfg["optim"]),
        dataloader_num_workers=int(train_cfg["dataloader_num_workers"]),
        gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
        remove_unused_columns=False,
        logging_first_step=True,
        seed=int(config["seed"]),
    )
    training_args_signature = inspect.signature(TrainingArguments.__init__)
    training_args_params = training_args_signature.parameters
    if "eval_strategy" in training_args_params:
        train_args_kwargs["eval_strategy"] = "steps"
    else:
        train_args_kwargs["evaluation_strategy"] = "steps"
    if "group_by_length" in training_args_params:
        train_args_kwargs["group_by_length"] = bool(train_cfg["group_by_length"])
    elif bool(train_cfg["group_by_length"]) and "train_sampling_strategy" in training_args_params:
        train_args_kwargs["train_sampling_strategy"] = "group_by_length"

    train_args_kwargs = {
        key: value
        for key, value in train_args_kwargs.items()
        if key in training_args_params
    }

    train_args = TrainingArguments(**train_args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    if config["export"]["save_adapter"]:
        adapter_dir = output_dir / "adapter"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

    if config["export"]["save_merged_16bit"]:
        merged_dir = output_dir / "merged_16bit"
        model.save_pretrained_merged(
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )

    if config["export"]["save_gguf"]:
        gguf_dir = output_dir / "gguf"
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method=config["export"]["gguf_quantization_method"],
        )


if __name__ == "__main__":
    main()
