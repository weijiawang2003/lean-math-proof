"""Runtime tactic policy backed by a trained sequence classifier.

The Policy class lazy-loads the checkpoint on first use, so importing this
module is always safe even when no checkpoint exists on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from actions import ACTIONS, load_action_space
from core_types import build_prompt

DEFAULT_CKPT_DIR = "clf_ckpt"


class Policy:
    """Lazy-loading tactic policy backed by a sequence classifier checkpoint.

    The model, tokenizer, and action space are loaded on the first call to
    ``choose_tactic``, not at construction time.  This makes it safe to
    instantiate (or import) without a checkpoint on disk.
    """

    def __init__(self, ckpt_dir: str = DEFAULT_CKPT_DIR):
        self._ckpt_dir = ckpt_dir
        self._model = None
        self._tokenizer = None
        self._actions: list[str] | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        ckpt = Path(self._ckpt_dir)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {ckpt}. "
                f"Train a model first with train_action_classifier.py."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(self._device)
        self._model.eval()

        # Load action space -------------------------------------------------
        action_file = ckpt / "action_space.json"
        if action_file.exists():
            self._actions = load_action_space(str(action_file))
        else:
            self._actions = list(ACTIONS)

        # Validate label-space alignment ------------------------------------
        n_labels = self._model.config.num_labels
        n_actions = len(self._actions)
        if n_labels != n_actions:
            raise ValueError(
                f"Action-space mismatch: model checkpoint has {n_labels} labels "
                f"but loaded action space has {n_actions} entries. "
                f"Ensure clf_ckpt/action_space.json matches the training config, "
                f"or retrain the model."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def actions(self) -> list[str]:
        self._ensure_loaded()
        return list(self._actions)

    @torch.inference_mode()
    def choose_tactic(self, state_pp: str, full_name: str = "") -> str:
        """Return the highest-scoring tactic for the given proof state."""
        return self.rank_tactics(state_pp, full_name, k=1)[0]

    @torch.inference_mode()
    def rank_tactics(self, state_pp: str, full_name: str = "", k: int = 5) -> list[str]:
        """Return the top-k tactics ranked by model confidence."""
        self._ensure_loaded()
        prompt = build_prompt(state_pp=state_pp, full_name=full_name)
        enc = self._tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self._device)
        logits = self._model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        top_k = min(k, len(self._actions))
        indices = torch.topk(probs, top_k).indices.tolist()
        return [self._actions[i] for i in indices]


# ======================================================================
# Backward-compatible module-level convenience function
# ======================================================================

_default_policy: Policy | None = None


def choose_tactic(state_pp: str, full_name: str = "") -> str:
    """Module-level convenience wrapper (lazy-loads default checkpoint)."""
    global _default_policy
    if _default_policy is None:
        _default_policy = Policy()
    return _default_policy.choose_tactic(state_pp, full_name)
