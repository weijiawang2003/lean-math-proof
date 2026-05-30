# RC2 Attribution Notes

## Credited delta: +5 (the headline)
Five theorems failed by **literal RC1** AND all four baselines
(simp / simp_all / aesop / classical<;>aesop), each closed by **single-shot**
`simp [Set.ite]`: `Set.ite_empty_right`, `Set.ite_right`, `Set.ite_empty`,
`Set.ite_empty_left`, `Set.ite_left`. Minimal-sufficient relabel: 5/5
TRUE_SET_ITE_SIMP_WIN, 0 baseline-duplicate, 0 parser-artifact. This is the only
figure used as the official RC2 improvement.

## Raw +18 observed in one benchmark — NOT the headline
The per-surface comparison summed to a raw +18 across 8 surfaces. This number is
**not** used as the headline because:
- It double-counts theorems shared across surfaces (e.g. `Set.ite_right` appears in
  known_wins, selected_failures, and the SF1 frontier subset).
- It folds in the 4 deferred depth-2 sequence wins (below).
- After de-duplication and single-shot attribution, the clean credited figure is +5.

## The +4 are SX3 depth-2 sequence candidates, not RC2 wins
`Set.ite_inter`, `Set.ite_inter_self`, `Set.ite_compl`, `Set.ite_inter_compl_self`
are closed deterministically by the deployable full-wrapper, but live forensic probes
show the mechanism is the **sequence** `simp [Set.ite] <;> aesop`: bare `aesop`,
bare `simp_all`, and single-shot `simp [Set.ite]` all fail; only the depth-2 sequence
closes them. In the full wrapper they close via `simp [Set.ite]` at step 1 then
`aesop` at step 2. They are deferred to a separate **SX3** depth-2 sequence-search
line, which must be validated on its own (literal-RC1 + minimal relabel) before any
credit. They are excluded from the RC2 official delta.

## No claim of theorem invention
RC2 invents no lemmas. Two consecutive SF2/SF3 investigations (Multiset singleton,
Set cluster) found **0 missing lemmas** — the frontier gaps are automation / routing /
short-sequence-composition gaps over EXISTING Mathlib lemmas. RC2's contribution is a
single narrow tactic-gate that composes an existing simp lemma set (`Set.ite`
unfolding) the base policy did not reach.

## Current frontier characterization
The live frontier is best described as **short proof-program mining** — finding the
1–2 step tactic sequences over existing lemmas that the deterministic policy fails to
compose or route to — not missing-lemma discovery. RC2 (single-shot) and the deferred
SX3 (depth-2 sequences) are the first two rungs of that ladder.
