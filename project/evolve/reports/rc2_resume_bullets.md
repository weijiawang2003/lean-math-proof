# RC2 Candidate — Resume Bullets (DRAFT)

- Designed and validated an additive, off-by-default extension to a production
  Lean/Mathlib proof-search stack (RC1 ⊕ `SET_ITE_SIMP`), composing one narrowly
  name-gated tactic (`simp [Set.ite]`) without modifying the protected production
  wrapper or router.
- Confirmed a **+5 credited delta over literal RC1** (not a proxy) with a rigorous
  attribution pipeline: literal-RC1 baseline, full-wrapper candidate eval, NS23
  minimal-sufficient relabel (5/5 true wins, 0 baseline-duplicate), and three
  hash-stable reproduction runs.
- Held the line on **0 regressions and 0 off-gate emissions** by exploiting a
  by-construction preservation argument (the gate filters only wrapper-added tactics;
  base-model output is never gated → candidate ≡ baseline on all non-target theorems).
- Diagnosed a full-wrapper integration subtlety: a gated battery action must live in
  `priority_templates` (emitted before the base policy), not `fallback_tactics` (which
  the per-state cap crowds out — empirically 1/5 vs 5/5).
- Ran live forensic probes that reclassified 4 ambiguous "perturbation" wins as clean
  **depth-2 sequence candidates** (`simp [Set.ite] <;> aesop` closes them; bare `aesop`
  and single-shot `simp [Set.ite]` do not), deferring them to a dedicated SX3
  sequence-search line rather than inflating the RC2 delta.
- Preserved canonical floors (demo_v1 11/15, nat_defs_medium 37/38, nat_defs_large_v5
  49/65) and produced a formal credited-delta ledger + draft release artifacts, gated
  on explicit owner approval before any production/README change.
