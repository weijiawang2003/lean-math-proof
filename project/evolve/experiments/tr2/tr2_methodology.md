# TR2 — Model-Guided Active Probing (methodology)

**Branch:** `sx3-depth2-sequence-search` · **Status:** experiment, off-by-default, no production change.

TR2 is the first **active-learning loop** built on top of the TR1 failure-to-action router. It asks a
single question:

> Is the TR1 router useful for **active data collection** — i.e. does using its predictions to *select*
> which RC2 failures / frontier cases to live-probe yield **more useful verified labels per LeanDojo
> probe** than a rule baseline or random selection?

**Success is not offline accuracy.** TR1 already measured that (and flagged it `PILOT_ONLY_NEEDS_MORE_DATA`
because of a 0.49 grouped-generalization gap). TR2 measures something different and downstream-relevant:
**whether the router earns its keep as a data-collection prioritizer.** The loop is:

```
TR1 router predictions
  → select cases (model-selected vs rule-selected vs random baseline)
  → confirm literal RC2 status (verified)
  → run cluster/prediction-appropriate probes (live, verified)
  → apply SX4 attribution (credit only true deltas)
  → update the training dataset with verified outcomes
  → compare selection strategies on useful-labels-per-probe
```

Everything is **verified-label discipline** (same as TR1/SF4/SX4): a probe outcome counts only after live
LeanDojo execution and SX4 attribution. No claimed win without a literal-production comparison.

---

## Definitions

**USEFUL_LABEL** — a live-probed case that produces one of:
- `TRUE_DELTA` — a gated probe genuinely beats literal RC2 (controls + depth-1 sub-controls fail), or
- `MISSING_BRIDGE_LEMMA_CANDIDATE` — confirmed failure whose residual is a likely-existing Mathlib
  bridge lemma (retrieval target, not a tactic), or
- `PROOF_SEARCH_DEPTH_GAP` — bare control closes the advanced state but RC2's bounded search missed it, or
- a **confirmed `NO_CHEAP_ACTION`** — verified that no cheap tactic/sequence helps (a real negative), or
- a `BASELINE_DUPLICATE` **with a clear control proof** — verified that a bare control (simp / simp_all /
  aesop / classical <;> aesop) closes it, i.e. a routing/depth gap rather than a missing capability, or
- **new negative evidence for a family** — a gate that fires but never closes (off-gate / over-fire
  evidence that demotes a candidate family).

A label that is merely *re-asserted* without a fresh verified observation is **not** newly useful; it can
still be a stability control.

**HIGH_VALUE_CASE** — a case that, if labelled, improves the dataset along an axis TR1's error analysis
flagged as weak:
- improves **label balance** (adds support to an under-represented class:
  `WX3_MULTISET_INDUCTION`, `MX2_TOFINSET_AESOP`, `PROOF_SEARCH_DEPTH_GAP`, or any non-Set namespace), or
- **resolves model uncertainty** (high predictive entropy / small top-1 margin), or
- sits in a cluster the router is least confident about.

**ACTIVE_LEARNING_GAIN** — useful labels produced **per LeanDojo probe budget spent**. This is the headline
metric of the strategy comparison: `useful_labels / live_probes` (and `true_delta / live_probes`),
broken down by selection strategy.

---

## Reuse discipline (compute honesty)

The candidate universe for TR2 is **already exhausted by prior stages**: every frontier theorem, every
confirmed RC2 failure, every RC2-solved case, and the entire TR1 active-learning list are already members
of the 57 TR1 training examples, and **SF4 already ran identical-config literal-RC2 confirmation and live
control/probe outcomes on all 27 confirmed failures (+ RC2 status on the 13 solved).**

TR2 therefore does **not** re-burn LeanDojo on settled cases. Instead:
- **RC2 confirmation** reuses SF4's identical-config (`rc2_production_wrapper.json`, `ns24_router.json`,
  `hybrid_evolved`, top-k 8, max-steps 8) literal-RC2 results as the oracle; it live-runs **only** cases
  with no SF4 record (provenance tagged `sf4_reused` vs `tr2_live`).
- **Live probes** reuse SF4's verified control/probe outcomes where the same tactic was already executed,
  and run a **genuine live increment** only for router-recommended probes SF4 did not cover (chiefly
  `PROOF_SEARCH_DEPTH_GAP` bounded sequences).

This keeps every outcome *verified* while spending no compute re-deriving settled results. Provenance is
recorded per outcome so the strategy comparison can distinguish reused-verified from freshly-probed.

**Expected honest finding.** Because the pool is fully pre-labelled and tiny (~40 unique theorems) and SF4
already found **0 cheap TRUE_DELTA**, TR2 cannot manufacture new true deltas. The legitimate questions it
*can* answer are (a) does model selection concentrate **more useful/diverse labels** than random/rule, and
(b) is the pool exhausted (→ the active-learning loop needs a genuinely fresh, multi-namespace frontier
before it can pay off). The decision space includes `INCONCLUSIVE_TOO_SMALL` precisely for this case.

## Guardrails

- No production routing change; `ns24_router.json` untouched.
- No edits to RC1/RC2 wrappers or release reports; no RC4.
- The router is **not** promoted; it is used only to *rank cases for probing*.
- All new files live under `project/evolve/experiments/tr2/`, `project/evolve/reports/tr2/`, `scripts/tr2_*`.
