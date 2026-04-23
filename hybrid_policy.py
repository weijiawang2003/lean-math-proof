"""Hybrid policy: generative model + fixed action space fallback.

Combines the creativity of a seq2seq/decoder-only generative model with
the reliability of a curated action space.  At each step:

  1. Generate k_gen tactics from the generative model (beam search)
  2. Rank k_fixed tactics from the fixed action space (if classifier exists)
     OR just include the top fixed tactics as-is
  3. Interleave: generative first, then fixed tactics not already generated
  4. Return deduplicated top-k

This gives best-of-both-worlds: the generative model can produce novel
tactics (e.g. `simp [Set.diff_eq, Set.compl_eq]`) while the fixed set
provides reliable fallbacks (`aesop`, `simp`, `tauto`).

Usage:
    pol = HybridPolicy(gen_ckpt_dir="gen_ckpt", action_space="search_v4")
    tactics = pol.rank_tactics(state_pp, full_name, k=10)
"""

from __future__ import annotations

from typing import Optional

from actions import get_action_space


class HybridPolicy:
    """Hybrid generative + fixed action space policy."""

    def __init__(
        self,
        gen_ckpt_dir: str = "gen_ckpt",
        clf_ckpt_dir: str = "",
        action_space: str = "search_v4",
        gen_weight: float = 0.7,
        fixed_weight: float = 0.3,
    ):
        self._gen_ckpt_dir = gen_ckpt_dir
        self._clf_ckpt_dir = clf_ckpt_dir
        self._action_space_name = action_space
        self._gen_weight = gen_weight
        self._fixed_weight = fixed_weight

        # Lazy-loaded
        self._gen_policy = None
        self._clf_policy = None
        self._fixed_actions: list[str] | None = None

    def _ensure_loaded(self) -> None:
        if self._gen_policy is not None:
            return

        from generative_policy import GenerativePolicy
        self._gen_policy = GenerativePolicy(ckpt_dir=self._gen_ckpt_dir)

        if self._clf_ckpt_dir:
            try:
                from policy import Policy
                self._clf_policy = Policy(ckpt_dir=self._clf_ckpt_dir)
                print(f"Hybrid: loaded classifier from {self._clf_ckpt_dir}")
            except Exception as exc:
                print(f"Hybrid: classifier load failed ({exc}), using fixed actions only")
                self._clf_policy = None

        self._fixed_actions = get_action_space(self._action_space_name)
        print(f"Hybrid policy: gen({self._gen_ckpt_dir}) + "
              f"{len(self._fixed_actions)} fixed actions ({self._action_space_name})")

    @property
    def model_type(self) -> str:
        return "hybrid"

    def rank_tactics(
        self,
        state_pp: str,
        full_name: str = "",
        k: int = 10,
    ) -> list[str]:
        """Generate top-k tactics by combining generative + fixed sources.

        Strategy:
          - Get k_gen = ceil(k * gen_weight) from generative model
          - Get remaining from fixed action space (classifier-ranked or raw)
          - Deduplicate, preserving generative-first ordering
        """
        self._ensure_loaded()

        k_gen = max(1, int(k * self._gen_weight + 0.5))
        k_fixed = k  # we'll truncate after dedup

        # 1. Generative tactics
        gen_tactics = self._gen_policy.rank_tactics(
            state_pp, full_name, k=k_gen
        )

        # 2. Fixed tactics (classifier-ranked if available, else raw order)
        if self._clf_policy:
            fixed_tactics = self._clf_policy.rank_tactics(
                state_pp, full_name, k=k_fixed
            )
        else:
            # Use a priority subset of the fixed actions
            # Top tactics that tend to work well, then the rest
            priority = [
                "aesop", "simp", "tauto", "omega", "ring", "norm_num",
                "simp_all", "exact?", "decide", "ext", "push_neg",
                "simp [Set.ext_iff]", "simp [Set.subset_def]",
                "simp [Set.mem_union]", "simp [Set.mem_inter_iff]",
                "simp [Set.diff_eq]", "simp [Set.mem_diff]",
                "simp [*]", "simp at *", "simpa",
                "constructor", "intro h", "rcases h with ⟨h1, h2⟩",
            ]
            # Priority first, then remaining fixed actions
            seen = set(priority)
            fixed_tactics = priority + [
                t for t in self._fixed_actions if t not in seen
            ]
            fixed_tactics = fixed_tactics[:k_fixed]

        # 3. Interleave: generative first, then fixed (deduped)
        result = []
        seen = set()
        for tac in gen_tactics:
            if tac not in seen:
                seen.add(tac)
                result.append(tac)
        for tac in fixed_tactics:
            if tac not in seen:
                seen.add(tac)
                result.append(tac)

        return result[:k]

    def choose_tactic(self, state_pp: str, full_name: str = "") -> str:
        """Return the single best tactic."""
        tactics = self.rank_tactics(state_pp, full_name, k=1)
        return tactics[0] if tactics else "sorry"
