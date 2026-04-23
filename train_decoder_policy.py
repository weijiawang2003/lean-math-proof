"""Train a decoder-only (causal LM) model to generate Lean tactics.

Complements train_tactic_generator.py (seq2seq/T5) with a GPT-2 style
causal language model.  The decoder-only approach is simpler and scales
better to larger models (GPT-2, CodeGen, StarCoder, etc.).

Training format:
  Input:  "PROOF_STATE: <state> TACTIC: <tactic>"
  The model learns to predict the tactic tokens given the proof state prefix.
  We mask the loss on the proof state tokens so only tactic prediction is trained.

Usage:
  python train_decoder_policy.py --data seq2seq_data.jsonl --output-dir dec_ckpt
  python train_decoder_policy.py --data seq2seq_data.jsonl --model gpt2 --epochs 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Special separator tokens
PROOF_STATE_PREFIX = "PROOF_STATE: "
TACTIC_PREFIX = " TACTIC: "
EOS_TOKEN = "<|endoftactic|>"


@dataclass
class DecoderExample:
    prompt: str
    tactic: str


class TacticDecoderDataset(Dataset):
    """Dataset for causal LM tactic generation.

    Each example is formatted as:
        PROOF_STATE: <state> TACTIC: <tactic><|endoftext|>

    Loss is only computed on the tactic tokens (prefix is masked with -100).
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 512,
    ):
        self.examples: List[DecoderExample] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        n_skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    self.examples.append(DecoderExample(
                        prompt=obj["prompt"],
                        tactic=obj["tactic"],
                    ))
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"[WARN] Skipping line {lineno}: {exc}")
                    n_skipped += 1

        if n_skipped:
            print(f"[WARN] Skipped {n_skipped} malformed lines")
        print(f"Loaded {len(self.examples)} decoder examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Build the full sequence: PROOF_STATE: <state> TACTIC: <tactic><eos>
        prefix = PROOF_STATE_PREFIX + ex.prompt + TACTIC_PREFIX
        full_text = prefix + ex.tactic + self.tokenizer.eos_token

        # Tokenize the full sequence
        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Create labels: mask the prefix tokens with -100
        # Tokenize just the prefix to find where tactic tokens start
        prefix_enc = self.tokenizer(
            prefix,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prefix_len = prefix_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        # Mask prefix tokens (we don't want to learn to predict the proof state)
        labels[:prefix_len] = -100
        # Mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train(
    data_path: str,
    model_name: str = "gpt2",
    output_dir: str = "dec_ckpt",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 5e-5,
    seed: int = 42,
    val_split: float = 0.1,
    max_length: int = 512,
) -> None:
    torch.manual_seed(seed)

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset
    full_dataset = TacticDecoderDataset(data_path, tokenizer, max_length)
    n_total = len(full_dataset)
    if n_total == 0:
        raise SystemExit(f"Dataset is empty: {data_path}")

    # Train/val split
    n_val = max(1, int(n_total * val_split)) if val_split > 0 else 0
    n_train = n_total - n_val
    if n_val > 0:
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)
        print(f"Split: {n_train} train, {n_val} val")
    else:
        train_ds = full_dataset
        val_ds = None
        print("No validation split")

    # Data collator (no MLM, this is causal LM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds else "no",
        load_best_model_at_end=bool(val_ds),
        seed=seed,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    # Handle transformers API differences
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    print(f"\nTraining for {epochs} epochs...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    unique_tactics = sorted(set(ex.tactic for ex in full_dataset.examples))
    tactic_hash = hashlib.sha256(json.dumps(unique_tactics).encode()).hexdigest()[:12]

    meta = {
        "model_type": "decoder",
        "base_model": model_name,
        "data_path": data_path,
        "total_examples": n_total,
        "train_examples": n_train,
        "val_examples": n_val,
        "unique_tactics_in_data": len(unique_tactics),
        "tactic_hash": tactic_hash,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "max_length": max_length,
    }
    meta_path = Path(output_dir) / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    vocab_path = Path(output_dir) / "seen_tactics.json"
    vocab_path.write_text(json.dumps(unique_tactics, indent=2), encoding="utf-8")

    print(f"\nModel saved to {output_dir}")
    print(f"  Unique tactics in training data: {len(unique_tactics)}")
    print(f"  Training metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Train decoder-only tactic generator.")
    parser.add_argument("--data", default="seq2seq_data.jsonl")
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace causal LM name (gpt2, codegen, etc.).")
    parser.add_argument("--output-dir", default="dec_ckpt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    train(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        val_split=args.val_split,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
