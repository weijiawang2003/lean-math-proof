import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
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
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.examples.append(Example(
                    prompt=obj["prompt"],
                    label=int(obj["label"]),
                ))
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
) -> None:
    actions = load_action_space(action_space_file) if action_space_file else get_action_space(action_space_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(actions),
    )

    dataset = SFTRawDataset(sft_path, tokenizer)
    print(f"Loaded {len(dataset)} SFT examples")
    print(f"Training label space size: {len(actions)}")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    action_space_out = str(Path(output_dir) / "action_space.json")
    save_action_space(action_space_out, actions)
    print("Model saved to", output_dir)
    print("Action space saved to", action_space_out)


def main():
    parser = argparse.ArgumentParser(description="Train action classifier from SFT dataset.")
    parser.add_argument("--sft-path", default="sft_dataset.jsonl")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="clf_ckpt")
    parser.add_argument("--action-space", default="core_v1", choices=list_action_spaces())
    parser.add_argument("--action-space-file", default="", help="Optional JSON file with {'actions': [...]}.")
    args = parser.parse_args()

    train(
        sft_path=args.sft_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        action_space_name=args.action_space,
        action_space_file=args.action_space_file,
    )


if __name__ == "__main__":
    main()
