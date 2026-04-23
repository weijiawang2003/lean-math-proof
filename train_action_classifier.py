import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from actions import get_action_space, list_action_spaces, load_action_space, save_action_space

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Example:
    prompt: str
    label: int


class SFTRawDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.examples: List[Example] = []
        n_skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    self.examples.append(Example(
                        prompt=obj["prompt"],
                        label=int(obj["label"]),
                    ))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    print(f"[WARN] Skipping malformed line {lineno}: {exc}")
                    n_skipped += 1
        if n_skipped:
            print(f"[WARN] Skipped {n_skipped} malformed line(s) in {path}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex.label, dtype=torch.long)
        return item


def train(
    sft_path: str,
    model_name: str,
    output_dir: str,
    action_space_name: str,
    action_space_file: str,
    seed: int = 42,
    val_split: float = 0.1,
) -> None:
    actions = load_action_space(action_space_file) if action_space_file else get_action_space(action_space_name)

    # Determinism
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(actions),
    )

    full_dataset = SFTRawDataset(sft_path, tokenizer)
    n_total = len(full_dataset)
    if n_total == 0:
        raise SystemExit(f"SFT dataset is empty: {sft_path}")

    print(f"Loaded {n_total} SFT examples")
    print(f"Training label space size: {len(actions)}")

    # ---- Train / Val split ------------------------------------------------
    n_val = max(1, int(n_total * val_split)) if val_split > 0 else 0
    n_train = n_total - n_val
    if n_val > 0:
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=generator)
        print(f"Split: {n_train} train, {n_val} val (seed={seed})")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print("No validation split (--val-split 0)")

    # ---- Training ---------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=bool(val_dataset),
        seed=seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ---- Save action space + training metadata ----------------------------
    action_space_out = str(Path(output_dir) / "action_space.json")
    save_action_space(action_space_out, actions)

    action_space_hash = hashlib.sha256(json.dumps(actions).encode()).hexdigest()[:12]
    meta = {
        "action_space_name": action_space_name if not action_space_file else f"file:{action_space_file}",
        "action_space_hash": action_space_hash,
        "num_labels": len(actions),
        "model_name": model_name,
        "sft_path": sft_path,
        "sft_rows": n_total,
        "train_rows": n_train,
        "val_rows": n_val,
        "seed": seed,
        "val_split": val_split,
    }
    meta_path = Path(output_dir) / "training_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Model saved to {output_dir}")
    print(f"Action space saved to {action_space_out} (hash={action_space_hash})")
    print(f"Training metadata saved to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Train action classifier from SFT dataset.")
    parser.add_argument("--sft-path", default="sft_dataset.jsonl")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="clf_ckpt")
    parser.add_argument("--action-space", default="core_v1", choices=list_action_spaces())
    parser.add_argument("--action-space-file", default="", help="Optional JSON file with {'actions': [...]}.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data to hold out for validation (0 to disable).")
    args = parser.parse_args()

    if args.val_split < 0 or args.val_split >= 1:
        raise SystemExit(f"--val-split must be in [0, 1), got {args.val_split}")

    train(
        sft_path=args.sft_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        action_space_name=args.action_space,
        action_space_file=args.action_space_file,
        seed=args.seed,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
