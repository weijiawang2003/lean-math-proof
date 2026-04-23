"""Train a seq2seq model to generate Lean tactics from proof states.

Uses CodeT5-small (60M params, designed for code) or any HuggingFace
encoder-decoder model. The model learns to produce tactic text given
a proof state prompt, removing the fixed action space ceiling.

Training data format (JSONL):
  {"prompt": "Theorem: Set.union_comm\n\nProof state:\n...", "tactic": "ext x; simp [or_comm]"}

Usage:
  python train_tactic_generator.py --data seq2seq_data.jsonl --output-dir gen_ckpt
  python train_tactic_generator.py --data seq2seq_data.jsonl --model t5-small --epochs 20
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
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


@dataclass
class Seq2SeqExample:
    prompt: str
    tactic: str


class TacticGenDataset(Dataset):
    """Dataset of (proof_state_prompt, tactic_text) pairs."""

    def __init__(self, path: str, tokenizer, max_src_len: int = 512, max_tgt_len: int = 128):
        self.examples: List[Seq2SeqExample] = []
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        n_skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    self.examples.append(Seq2SeqExample(
                        prompt=obj["prompt"],
                        tactic=obj["tactic"],
                    ))
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"[WARN] Skipping line {lineno}: {exc}")
                    n_skipped += 1

        if n_skipped:
            print(f"[WARN] Skipped {n_skipped} malformed lines")
        print(f"Loaded {len(self.examples)} seq2seq examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Encode source (proof state prompt)
        src = self.tokenizer(
            ex.prompt,
            max_length=self.max_src_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Encode target (tactic text)
        # Note: as_target_tokenizer() was removed in newer transformers.
        # For T5/CodeT5, source and target share the same tokenizer,
        # so we can use text_target parameter or just call directly.
        tgt = self.tokenizer(
            text_target=ex.tactic,
            max_length=self.max_tgt_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels": tgt["input_ids"].squeeze(0),
        }

        # Replace padding token ids in labels with -100 so they're ignored in loss
        labels = item["labels"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        item["labels"] = labels

        return item


def train(
    data_path: str,
    model_name: str = "t5-small",
    output_dir: str = "gen_ckpt",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 5e-5,
    seed: int = 42,
    val_split: float = 0.1,
    max_src_len: int = 512,
    max_tgt_len: int = 128,
) -> None:
    torch.manual_seed(seed)

    print(f"Loading tokenizer and model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except TypeError:
        # Some models (e.g. codet5) have added_tokens.json incompatible with
        # newer transformers.  Fall back to loading without added tokens.
        print(f"[WARN] Tokenizer for {model_name} failed — falling back to t5-small")
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset
    full_dataset = TacticGenDataset(data_path, tokenizer, max_src_len, max_tgt_len)
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

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Training args
    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        seed=seed,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    # Newer transformers renamed 'tokenizer' → 'processing_class'
    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Seq2SeqTrainer(
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
    # Collect unique tactics from dataset for reference
    unique_tactics = sorted(set(ex.tactic for ex in full_dataset.examples))
    tactic_hash = hashlib.sha256(json.dumps(unique_tactics).encode()).hexdigest()[:12]

    meta = {
        "model_type": "seq2seq",
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
        "max_src_len": max_src_len,
        "max_tgt_len": max_tgt_len,
    }
    meta_path = Path(output_dir) / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Also save the tactic vocabulary for reference
    vocab_path = Path(output_dir) / "seen_tactics.json"
    vocab_path.write_text(json.dumps(unique_tactics, indent=2), encoding="utf-8")

    print(f"\nModel saved to {output_dir}")
    print(f"  Unique tactics in training data: {len(unique_tactics)}")
    print(f"  Training metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Train seq2seq tactic generator.")
    parser.add_argument("--data", default="seq2seq_data.jsonl", help="Path to seq2seq JSONL dataset.")
    parser.add_argument("--model", default="t5-small",
                        help="HuggingFace model name (encoder-decoder).")
    parser.add_argument("--output-dir", default="gen_ckpt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-src-len", type=int, default=512)
    parser.add_argument("--max-tgt-len", type=int, default=128)
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
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    )


if __name__ == "__main__":
    main()
