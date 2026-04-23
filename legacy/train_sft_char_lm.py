import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. 配置
# =========================

JSONL_PATH = "traces.jsonl"      # 你刚刚生成的文件
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
NUM_EPOCHS = 20
LR = 1e-3
SAVE_PATH = "char_lm_ckpt.pt"

# 文本模板：把 state 和 tactic 拼成一个串
def make_text_example(state_pp: str, tactic: str) -> str:
    return state_pp + "\n\nTACTIC:\n" + tactic


# =========================
# 2. 数据集 & 字符词表
# =========================

PAD_TOKEN = "<pad>"

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class CharDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples: List[str] = []

        # 1) 从 JSONL 里读出所有文本样本
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                state_pp = obj["state_pp"]
                tactic = obj["tactic"]
                txt = make_text_example(state_pp, tactic)
                self.samples.append(txt)

        # 2) 建立字符级 vocab
        chars = set()
        chars.add(PAD_TOKEN)
        for txt in self.samples:
            for ch in txt:
                chars.add(ch)

        itos = sorted(list(chars))
        stoi = {ch: i for i, ch in enumerate(itos)}
        self.vocab = Vocab(stoi=stoi, itos=itos)

        # 3) 把所有样本预编码成 id 序列
        self.encoded: List[List[int]] = [self.vocab.encode(t) for t in self.samples]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        ids = self.encoded[idx]
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor], pad_id: int):
    # batch: [seq_len_i] 的一堆 1D tensor
    max_len = max(x.size(0) for x in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    target_ids = torch.full((B, max_len), pad_id, dtype=torch.long)

    for i, seq in enumerate(batch):
        L = seq.size(0)
        input_ids[i, :L] = seq
        # language model 目标：预测下一个字符
        target_ids[i, :L-1] = seq[1:]
        # 最后一个 token 的 target 保持 pad，无需预测

    return input_ids, target_ids


# =========================
# 3. 模型：字符级 GRU 语言模型
# =========================

class CharGRULM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        # input_ids: [B, T]
        x = self.embedding(input_ids)      # [B, T, E]
        out, _ = self.gru(x)              # [B, T, H]
        logits = self.fc_out(out)         # [B, T, V]
        return logits


# =========================
# 4. 训练循环
# =========================

def train():
    dataset = CharDataset(JSONL_PATH)
    vocab = dataset.vocab

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab.pad_id),
    )

    model = CharGRULM(
        vocab_size=vocab.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Device: {DEVICE}")
    print(f"Num samples: {len(dataset)}, Vocab size: {vocab.vocab_size}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(DEVICE)   # [B, T]
            target_ids = target_ids.to(DEVICE) # [B, T]

            optimizer.zero_grad()
            logits = model(input_ids)          # [B, T, V]

            # reshape for cross-entropy: [B*T, V] vs [B*T]
            B, T, V = logits.shape
            loss = criterion(
                logits.view(B * T, V),
                target_ids.view(B * T),
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mask = (target_ids != vocab.pad_id)
                n_tokens = mask.sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

        ppl = math.exp(total_loss / max(1, total_tokens))
        print(f"Epoch {epoch:02d} | loss/token={total_loss/total_tokens:.4f} | ppl={ppl:.2f}")

    # 保存模型和 vocab
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_stoi": vocab.stoi,
            "vocab_itos": vocab.itos,
        },
        SAVE_PATH,
    )
    print(f"Saved checkpoint to {SAVE_PATH}")


# =========================
# 5. 推理：给 state_pp 生成 tactic
# =========================

@torch.inference_mode()
def generate_tactic(
    state_pp: str,
    ckpt_path: str = SAVE_PATH,
    max_new_tokens: int = 128,
) -> str:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    vocab = Vocab(stoi=ckpt["vocab_stoi"], itos=ckpt["vocab_itos"])

    model = CharGRULM(
        vocab_size=vocab.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prefix = make_text_example(state_pp, tactic="")  # 这里只放 state + "TACTIC:" 前缀
    # 实际上 make_text_example 会生成 `state_pp + "\n\nTACTIC:\n"`，后面 tactic 部分为空

    input_ids = torch.tensor(vocab.encode(prefix), dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :]   # [1, V]
        probs = torch.softmax(logits, dim=-1)
        # 用 argmax 而不是随机采样
        next_id = torch.argmax(probs, dim=-1, keepdim=True)  # [1, 1]

        input_ids = torch.cat([input_ids, next_id], dim=1)

        if vocab.itos[next_id.item()] == "\n":
            break

    # 把生成部分从 prefix 后面截出来
    full_text = vocab.decode(input_ids[0].tolist())
    # full_text = state_pp + "\n\nTACTIC:\n" + generated
    if "\n\nTACTIC:\n" in full_text:
        tactic_part = full_text.split("\n\nTACTIC:\n", 1)[1]
    else:
        tactic_part = full_text

    # 只取第一行作为 tactic
    tactic_line = tactic_part.splitlines()[0]
    return tactic_line


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "gen"], default="train")
    parser.add_argument("--state-file", type=str, default=None,
                        help="File containing a state_pp to test generation.")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        if args.state_file is None:
            raise SystemExit("Please provide --state-file when mode=gen")
        with open(args.state_file, "r", encoding="utf-8") as f:
            state_pp = f.read()
        tactic = generate_tactic(state_pp)
        print("Generated tactic:")
        print(tactic)
