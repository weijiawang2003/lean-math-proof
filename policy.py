import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from actions import ACTIONS

CKPT_DIR = "clf_ckpt"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR).to(DEVICE)
model.eval()

@torch.inference_mode()
def choose_tactic(state_pp: str, full_name: str = "") -> str:
    prompt = f"Theorem: {full_name}\n\nProof state:\n{state_pp}\n"
    enc = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(DEVICE)
    logits = model(**enc).logits  # [1, num_actions]
    probs = torch.softmax(logits, dim=-1)
    idx = torch.argmax(probs, dim=-1).item()
    return ACTIONS[idx]

