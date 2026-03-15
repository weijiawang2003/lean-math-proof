import json
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from actions import ACTIONS


SFT_PATH   = "sft_dataset.jsonl"
MODEL_NAME = "distilbert-base-uncased"   # 先用小模型，把 pipeline 跑通
OUTPUT_DIR = "clf_ckpt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


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


def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ACTIONS),
    )

    dataset = SFTRawDataset(SFT_PATH, tokenizer)
    print(f"Loaded {len(dataset)} SFT examples")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
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
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved to", OUTPUT_DIR)


if __name__ == "__main__":
    train()
