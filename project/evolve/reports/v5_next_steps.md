# v5 → v6: concrete next steps

This document is a short, actionable list of next steps. Each item
is sized for a single PR / single research cycle.

## Tier 1 — quick wins (≤ 1 day each)

### NS1. Fix the wave 5 finding: per-slot specificity ordering

The wave 5 v5-31 result showed that putting generic omega-omega
templates first in the iff slot regresses 4 wins. The wrapper's
"first-non-erroring-wins" semantics propagates within the slot.

Fix: when emitting from priority_templates, the wrapper should
classify each template as "specific" (uses `{hyp_*}` placeholders
or names a non-trivial Mathlib lemma) vs "generic" (omega/simp_all
fallback), and emit all specific templates before any generic ones,
regardless of declared order.

Effort: ~50 LoC in `strategy_wrapper.py`. No genome schema change.
Expected: closes the per-slot ordering risk; makes hand-curation safer.

### NS2. Add `Nat.add_mod_eq_ite` via hand-coded multi-step skeleton

See `v5_failure_deep_dive_add_mod_eq_ite.md`. A single hand-coded
priority template chaining `rw [Nat.add_mod] ; split_ifs with h ; ...`
with the correct asymmetric inner tactics would close this one.

Effort: ~20 LoC. Adds one new template to the genome.
Expected: 32/38.

### NS3. Lemma-name retrieval pass for unsolved theorems

Several of v5's failures stalled on "I couldn't find the right
Mathlib lemma name." A 30-minute targeted search in the Mathlib
source for each failing theorem (Nat.div_le_div_right,
Nat.dvd_iff_div_mul_eq, Nat.pow_lt_pow_iff_left) would likely
surface candidates.

Effort: human-driven. Each candidate becomes a new priority template.
Expected: 1-3 more closures.

## Tier 2 — structural (1-2 weeks)

### NS4. v6 skeleton-bag refactor

Implement `evolve/skeleton_bag.py` per `v5_alphaevolve_architecture.md`.
A skeleton is a typed object with (name, shape, template_body,
slot_vocabulary) fields. The wrapper iterates over the bag
shape-keyed, instantiating each skeleton's slots against the
state's vars/hyps. Old flat lists (`tactic_templates`,
`family_tactics`, etc.) become syntactic sugar over the bag.

Effort: ~500 LoC + tests. Backward-compatible via skeleton-bag-from-
old-config builder.
Expected: cleaner architecture; mutator can finally add new
skeletons rather than just permute existing ones.

### NS5. Two-tier mutator

After NS4, extend `evolve/mutator.py` with:
  - outer-tier ops: add/remove skeleton from bag; change skeleton
    shape gate.
  - inner-tier ops: mutate slot vocabulary; reorder slot entries.
Outer-tier should be 10% of mutations (high-impact, low-frequency).

Effort: ~300 LoC. Reuses existing `mutator.py` for inner-tier.
Expected: enables genuine AlphaEvolve search, not just slot-content
walk.

### NS6. Cross-run archive

Implement `project/evolve/archive/skeletons.jsonl` per the
architecture doc. Each row: (skeleton-name, slots-filled, theorem,
first_seen_run, wins, regressions). The mutator seeds new candidates
from archived top-N skeletons.

Effort: ~150 LoC. JSONL only; no DB needed.
Expected: cumulative learning across runs. The 5 priority_templates
wins from tonight would be archive seeds for the next run.

## Tier 3 — Learn step (3-5 days)

### NS7. Build the v5 training data and fine-tune gen_v5+1

The pipeline ships in `scripts/build_v5_training_data.py`. Tonight
it produced 157 (state, tactic) pairs across 29 theorems with
the 3 priority-templates wins held out.

  1. Train gen_v5+1 on (original gen_v5 dataset) ∪ (v5 evolve dataset).
  2. Evaluate raw gen_v5+1 on nat_defs_medium without the wrapper.
  3. Success metric: gen_v5+1 raw proves `Nat.div_lt_one_iff` or any
     of the priority-templates wins natively.
  4. If success: the wrapper's priority_templates slot becomes
     redundant for that theorem; we have a true Learn step.
  5. If failure: the wrapper remains load-bearing and the v5 wins
     don't generalize at the model level. (This is also a useful
     finding — it would tell us the T5-small model capacity is the
     bottleneck, not the data.)

Effort: ~1 day for the training run + ~30 min for the eval.
Expected: either redundancy (good) or capacity-limit confirmation
(also good).

### NS8. Hold-out: cross-domain eval after training

After NS7, evaluate gen_v5+1 raw on `nat_defs_large_v5` (the
30 unseen theorems) AND on `demo_v1` (Set/Finset domain). The
honest cross-domain transfer test.

Effort: ~30 min for two evals.
Expected: tells us whether the learning generalized to other
arithmetic theorems and whether it leaked anything bad to Set goals.

## Tier 4 — far horizon

### NS9. LLM-driven mutator

The v5 mutator is heuristic. An LLM could propose new priority
templates by reading the failing trace and suggesting alternative
Mathlib lemma forms. Implementation: replace `evolve/mutator.py`'s
deterministic ops with a small LLM call seeded with the failing
trace.

Effort: ~1 week including evaluation infrastructure for "did the
LLM suggestion actually close a theorem?"
Expected: closes the lemma-name lookup bottleneck (Claim 4 in
`v5_central_findings.md`).

### NS10. Move off nat_defs_medium

nat_defs_medium has done its job. The remaining 7 failures need
either env upgrades (Mathlib refresh, `nlinarith` availability) or
structural code changes. The next theorem set should test cross-
domain transfer in earnest:

  - `curriculum_all` (31 thm) — mostly Set, baseline near 100%.
    Useful for measuring whether wrapper additions break Set wins.
  - A fresh `Mathlib/Data/List/Basic.lean` slice — totally unseen
    domain. Honest transfer test.
  - `Mathlib/Data/Finset/Basic.lean` deep slice — partially seen.

## Priority for the v5 → v6 transition

Order I'd recommend:

  1. **NS3** (lemma-name retrieval pass) — cheap, immediate ≤3 new wins.
  2. **NS1** (per-slot specificity) — safety fix; no genome change.
  3. **NS4 + NS5** (skeleton bag + two-tier mutator) — the main v6.
  4. **NS6** (archive) — once v6 has the skeleton type.
  5. **NS7 + NS8** (Learn step) — parallel track.
  6. **NS9** (LLM mutator) — once the deterministic mutator's
     ceiling is well-understood.
  7. **NS10** (new domain) — final shakedown.

NS1-NS3 are quick-win one-day items. NS4-NS6 are the real v6.
NS7-NS8 connect to the Learn step. NS9-NS10 are the long-arc work.
