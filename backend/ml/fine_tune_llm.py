"""Fine-tune a language model using LoRA/PEFT on a CSV dataset.

This script loads a dataset with ``input`` and ``output`` columns, combines
them into a dialogue-style training text and performs LoRA fine-tuning using
HuggingFace ``transformers`` and ``peft``. The resulting weights and metrics
are stored under ``artifacts/<version>/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_dataset(path: str) -> Dataset:
    """Load dataset from ``path`` and return a HF ``Dataset``."""
    df = pd.read_csv(path)
    if "input" not in df.columns or "output" not in df.columns:
        raise ValueError("CSV must contain 'input' and 'output' columns")
    df["text"] = "User: " + df["input"].astype(str) + "\nAssistant: " + df["output"].astype(str)
    return Dataset.from_pandas(df[["text"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument("data", help="Path to dataset CSV with columns 'input' and 'output'")
    parser.add_argument("--base_model", default="gpt2", help="Base model name")
    parser.add_argument("--version", default="v1", help="Version string for artifacts")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    args = parser.parse_args()

    dataset = load_dataset(args.data)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="artifacts/tmp",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        eval_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    eval_result = trainer.evaluate()
    perplexity = float(torch.exp(torch.tensor(eval_result["eval_loss"])))

    version_dir = Path("artifacts") / args.version
    version_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(version_dir)
    tokenizer.save_pretrained(version_dir)
    with open(version_dir / "metrics.txt", "w") as f:
        f.write(f"Perplexity: {perplexity}\n")


if __name__ == "__main__":
    main()
