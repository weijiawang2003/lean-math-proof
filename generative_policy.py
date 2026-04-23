"""Generative tactic policy backed by a seq2seq or decoder-only model.

Supports two architectures:
  - seq2seq (T5, CodeT5): encoder-decoder, generates tactic from proof state
  - decoder (GPT-2, CodeGen): causal LM, generates tactic after proof state prefix

Architecture is auto-detected from training_meta.json in the checkpoint dir.

Uses beam search decoding to produce multiple candidate tactics,
ranked by model confidence (sequence log-probability).

Also provides PremiseAugmentedPolicy: wraps any GenerativePolicy to prepend
retrieved premises to the prompt at inference time (matching the training
format from build_premise_augmented_dataset.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from core_types import build_prompt

DEFAULT_GEN_CKPT = "gen_ckpt"

# Must match train_decoder_policy.py
PROOF_STATE_PREFIX = "PROOF_STATE: "
TACTIC_PREFIX = " TACTIC: "


class GenerativePolicy:
    """Lazy-loading generative tactic policy.

    Generates tactic text from proof state using beam search decoding.
    Compatible with the same rollout interface as policy.Policy.

    Auto-detects model type (seq2seq vs decoder) from checkpoint metadata.
    """

    def __init__(self, ckpt_dir: str = DEFAULT_GEN_CKPT):
        self._ckpt_dir = ckpt_dir
        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._meta: dict | None = None
        self._model_type: str = "seq2seq"  # default, auto-detected on load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        ckpt = Path(self._ckpt_dir)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Generative checkpoint not found: {ckpt}. "
                f"Train a model first with train_tactic_generator.py or train_decoder_policy.py."
            )

        # Load metadata to determine model type
        meta_path = ckpt / "training_meta.json"
        if meta_path.exists():
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._model_type = self._meta.get("model_type", "seq2seq")
        else:
            self._meta = {}
            self._model_type = "seq2seq"

        self._tokenizer = AutoTokenizer.from_pretrained(str(ckpt))

        if self._model_type == "decoder":
            self._model = AutoModelForCausalLM.from_pretrained(str(ckpt)).to(self._device)
            # GPT-2 needs pad token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        else:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to(self._device)

        self._model.eval()

        print(f"Loaded generative policy from {ckpt} (type={self._model_type})")
        print(f"  Base model: {self._meta.get('base_model', '?')}")
        print(f"  Unique tactics in training: {self._meta.get('unique_tactics_in_data', '?')}")

    @property
    def model_type(self) -> str:
        return self._model_type

    @torch.inference_mode()
    def _generate_seq2seq(
        self,
        prompt: str,
        num_samples: int,
        max_length: int,
        temperature: float,
        num_beams: int,
    ) -> list[str]:
        """Generate tactics using seq2seq (T5/CodeT5) model."""
        enc = self._tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self._device)

        if num_beams > 0:
            outputs = self._model.generate(
                **enc,
                max_length=max_length,
                num_beams=max(num_beams, num_samples),
                num_return_sequences=num_samples,
                early_stopping=True,
                no_repeat_ngram_size=0,
            )
        else:
            outputs = self._model.generate(
                **enc,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=num_samples,
            )

        tactics = []
        seen = set()
        for seq in outputs:
            text = self._tokenizer.decode(seq, skip_special_tokens=True).strip()
            if text and text not in seen:
                seen.add(text)
                tactics.append(text)
        return tactics

    @torch.inference_mode()
    def _generate_decoder(
        self,
        prompt: str,
        num_samples: int,
        max_length: int,
        temperature: float,
        num_beams: int,
    ) -> list[str]:
        """Generate tactics using decoder-only (GPT-2) model."""
        # Build the prefix that the model was trained on
        prefix = PROOF_STATE_PREFIX + prompt + TACTIC_PREFIX

        enc = self._tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
            max_length=max_length - 64,  # leave room for tactic generation
        ).to(self._device)

        prefix_len = enc["input_ids"].shape[1]

        if num_beams > 0:
            outputs = self._model.generate(
                **enc,
                max_new_tokens=64,
                num_beams=max(num_beams, num_samples),
                num_return_sequences=num_samples,
                early_stopping=True,
                no_repeat_ngram_size=0,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        else:
            outputs = self._model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=num_samples,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        tactics = []
        seen = set()
        for seq in outputs:
            # Decode only the generated part (after prefix)
            generated = seq[prefix_len:]
            text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
            if text and text not in seen:
                seen.add(text)
                tactics.append(text)
        return tactics

    def generate_tactics(
        self,
        state_pp: str,
        full_name: str = "",
        num_samples: int = 8,
        max_length: int = 128,
        temperature: float = 1.0,
        num_beams: int = 0,
    ) -> list[str]:
        """Generate candidate tactic strings for the given proof state.

        If num_beams > 0, uses beam search (deterministic, higher quality).
        Otherwise, uses sampling with temperature (diverse but noisier).

        Returns deduplicated list of up to num_samples tactics.
        """
        self._ensure_loaded()
        prompt = build_prompt(state_pp=state_pp, full_name=full_name)

        if self._model_type == "decoder":
            return self._generate_decoder(
                prompt, num_samples, max_length, temperature, num_beams
            )
        else:
            return self._generate_seq2seq(
                prompt, num_samples, max_length, temperature, num_beams
            )

    def rank_tactics(
        self,
        state_pp: str,
        full_name: str = "",
        k: int = 5,
    ) -> list[str]:
        """Generate top-k tactics using beam search.

        This method matches the interface of policy.Policy.rank_tactics(),
        making it a drop-in replacement for eval_rollout_all.py.
        """
        return self.generate_tactics(
            state_pp=state_pp,
            full_name=full_name,
            num_samples=k,
            num_beams=k * 2,  # use wider beam for better diversity
        )

    def choose_tactic(self, state_pp: str, full_name: str = "") -> str:
        """Return the single best tactic (beam search, top-1)."""
        tactics = self.rank_tactics(state_pp, full_name, k=1)
        if not tactics:
            return "sorry"  # fallback -- always valid in Lean (admits the goal)
        return tactics[0]


class PremiseAugmentedPolicy:
    """Wraps a GenerativePolicy to prepend retrieved premises at inference.

    This matches the training format from build_premise_augmented_dataset.py:
    the model was trained with "Relevant premises: ..." prepended to prompts,
    so at inference time we must do the same.

    The premise retriever is loaded lazily and uses the same index built
    during dataset construction.

    Usage:
        pol = PremiseAugmentedPolicy(
            ckpt_dir="project/gen_ckpt_v6",
            premise_index_path="project/premise_index.json",
        )
        tactics = pol.rank_tactics(state_pp, full_name, k=8)
    """

    def __init__(
        self,
        ckpt_dir: str = DEFAULT_GEN_CKPT,
        premise_index_path: str = "project/premise_index.json",
        traces_path: str = "project/all_traces.jsonl",
        max_premises: int = 10,
        k_retrieved: int = 15,
    ):
        self._inner = GenerativePolicy(ckpt_dir=ckpt_dir)
        self._premise_index_path = premise_index_path
        self._traces_path = traces_path
        self._max_premises = max_premises
        self._k_retrieved = k_retrieved
        self._retriever = None

    def _ensure_retriever(self) -> None:
        if self._retriever is not None:
            return
        from premise_retriever import PremiseRetriever
        self._retriever = PremiseRetriever()
        if Path(self._premise_index_path).exists():
            self._retriever.load_index(self._premise_index_path)
            print(f"[PremiseAugmentedPolicy] Loaded premise index from {self._premise_index_path}")
        else:
            self._retriever.build_index_from_traces(self._traces_path)

    def _augment_prompt(self, state_pp: str, full_name: str) -> str:
        """Build premise-augmented prompt matching training format."""
        self._ensure_retriever()
        premises = self._retriever.retrieve(state_pp, k=self._k_retrieved)
        truncated = premises[:self._max_premises]

        base = build_prompt(state_pp=state_pp, full_name=full_name)
        if truncated:
            prefix = "Relevant premises: " + ", ".join(truncated) + "\n"
            return prefix + base
        return base

    @property
    def model_type(self) -> str:
        return "premise_augmented"

    def generate_tactics(
        self,
        state_pp: str,
        full_name: str = "",
        num_samples: int = 8,
        max_length: int = 128,
        temperature: float = 1.0,
        num_beams: int = 0,
    ) -> list[str]:
        """Generate tactics with premise-augmented prompt."""
        self._inner._ensure_loaded()
        prompt = self._augment_prompt(state_pp, full_name)

        if self._inner._model_type == "decoder":
            return self._inner._generate_decoder(
                prompt, num_samples, max_length, temperature, num_beams
            )
        else:
            return self._inner._generate_seq2seq(
                prompt, num_samples, max_length, temperature, num_beams
            )

    def rank_tactics(
        self,
        state_pp: str,
        full_name: str = "",
        k: int = 5,
    ) -> list[str]:
        """Generate top-k tactics with premise-augmented prompt."""
        return self.generate_tactics(
            state_pp=state_pp,
            full_name=full_name,
            num_samples=k,
            num_beams=k * 2,
        )

    def choose_tactic(self, state_pp: str, full_name: str = "") -> str:
        """Return the single best tactic."""
        tactics = self.rank_tactics(state_pp, full_name, k=1)
        return tactics[0] if tactics else "sorry"
