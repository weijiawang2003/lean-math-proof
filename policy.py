import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from actions import ACTIONS
from core_types import build_prompt

CKPT_DIR = "clf_ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR).to(DEVICE)
model.eval()


@torch.inference_mode()
def choose_tactic(state_pp: str, full_name: str = "") -> str:
    prompt = build_prompt(state_pp=state_pp, full_name=full_name)
    enc = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(DEVICE)
    logits = model(**enc).logits
    idx = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).item()
    return ACTIONS[idx]
