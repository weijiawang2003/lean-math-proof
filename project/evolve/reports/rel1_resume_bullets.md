# REL1 — resume bullets

Reusable bullets describing the project at three levels of detail.

## Short (3)

- Built an automated Lean 4 / Mathlib theorem-proving system (routed generative
  policy + evolved deterministic search wrapper) that proves **+15 more theorems
  than the strong baseline with zero regressions**.
- Designed a **symbolic-action abstraction** that turns state-dependent Lean
  tactics into stable, learnable labels — the key to gains on quotient types.
- Evaluated end-to-end against **live LeanDojo** (real Lean kernel), shipping a
  reproducible release-candidate config with a full benchmark + ablation.

## Medium (3)

- Architected a layered proof-search stack — NS9 base wrapper + namespace-gated
  symbolic actions + targeted tactic fallbacks — and consolidated only the
  provably-safe gains into a production config (**RC1: +15 vs baseline, 0
  regressions, 0 off-gate emissions**, canonical floors preserved).
- Introduced a typed **symbolic-action layer** (`induction v using
  Multiset.induction_on <;> simp_all`, `cases v <;> simp`) that instantiates the
  variable from the live proof state, unlocking a fresh quotient-type namespace
  (Multiset) where direct fine-tuning had saturated.
- Ran a rigorous experimental program (10+ arcs) with **minimal-tactic relabeling**
  to prevent false credit, establishing exactly when SFT, wrapper, symbolic, and
  learned-selector techniques each pay off.

## Technical (3)

- Implemented and validated **WX3 Multiset induction oracle (+12)** and a
  **narrow `Set.Finite` `aesop` fallback (+3)** as additive, namespace-gated
  deltas on disjoint namespaces, proving 0 negative interaction via component
  ablation and 0 off-gate emissions via a static gate-logic check over theorem
  names.
- Trained a **learned symbolic-action predictor** (TF-IDF char_wb + balanced
  logistic regression over the proof state; CV top-1 0.90) that cleared a
  held-out promotion bar (retain 53.8%, 0 regressions, 0 non-namespace
  false-fires) — but showed the deterministic oracle dominates when emission is
  free, so kept it off by default.
- Prototyped **depth-2 symbolic sequence search** and demonstrated via offline
  trace replay that the existing best-first search already subsumes it (0 net
  wins), avoiding a costly dead-end; drove all conclusions from live LeanDojo
  evals over fresh, deduplicated Mathlib theorem frontiers.
