# Evaluation report — nat_defs_medium / hybrid_evolved + v4.1 retrieval

**Run id**: `evolve-20260521-233937-cf2370`
**Metrics**: `project/evolve/runs/evolve-20260521-233937-cf2370/eval/seed-baseline/eval-38e7aaf6/metrics.json`
**Checkpoint**: `project/models/gen_v5` (unchanged from v3.6)
**Top-k**: 8, **Max-steps**: 8, **Decode**: beam
**Wallclock**: 5m 22s (eval subprocess 18:39:37 → 18:44:59), ~8.5s/theorem

## Summary

- **Proved**: **25/38** (65.8%) — **identical to v3.6** (errored 6, exhausted 7, skipped 0)
- **Δ over v3.6**: 0 gains, 0 regressions. All 25 v3.6 wins reproduced with the same winning tactic and origin.
- **Δ over gen_v5 baseline**: +22 theorems (unchanged).

Same model. Same Lean. Same fallback/family/deny-list ordering as v3.6, plus the v4.1 premise-retrieval layer added between family and generic entries for `div`-family theorems only.

## Success criteria

| Criterion | Met? | Detail |
|---|---|---|
| ≥ 25/38 preserved | ✓ | 25/38 (matches v3.6) |
| Zero regressions on previously-solved theorems | ✓ | Same 25 wins, same winning tactic per theorem |
| No new `DojoCrashError` | ✓ | Denied tactics count unchanged at 8 (per-theorem deny-list intact) |
| Runtime remains acceptable | ✓ | 5m 22s vs v3.6's ~3m 28s; +1m 54s overhead all spent in retrieval attempts on the 6 div theorems |
| ≥ 1 new div-family theorem solved (*ideal*) | ✗ | 0 new div wins. Retrieval **advanced state** on 2 of 6 div theorems but never closed |

The infrastructure (retriever helper, wrapper integration, origin tagging, aggregated metrics) is in place and behaves correctly. v4.1 ships the **plumbing**; the **content** of the seeded div-premise list is the limiting factor.

## v4.1 retrieval aggregates

| Metric | Value |
|---|---|
| `retrieved_premise_activation_count` | 6 (all 6 div theorems activated retrieval) |
| `retrieved_premise_attempt_count` | 783 (retrieved-tactic Lean roundtrips across all 6 theorems) |
| `retrieved_premise_advanced_count` | 6 (state-advancing transitions: 5 on `Nat.div_lt_one_iff`, 1 on `Nat.dvd_iff_div_mul_eq`) |
| `retrieved_premise_proved_count` | 0 |
| `retrieved_premise_wins` | `[]` |

The 6 state advances confirm retrieved tactics did real Lean work — they're not just bouncing off type errors. They just don't close.

## Div-family per-theorem result

| Theorem | v3.6 status | v4.1 status | retrieval attempts | advances |
|---|---|---|---|---|
| `Nat.div_le_div_right` | ERROR (step 3) | EXHAUSTED (step 8) | 204 | 0 |
| `Nat.div_lt_iff_lt_mul'` | ERROR (step 2) | EXHAUSTED (step 8) | 238 | 0 |
| `Nat.div_lt_one_iff` | ERROR (step 4) | EXHAUSTED (step 8) | 170 | **5** |
| `Nat.div_pos` | ERROR (step 3) | ERROR (step 3) | 34 | 0 |
| `Nat.div_pos_iff` | ERROR (step 4) | ERROR (step 4) | 34 | 0 |
| `Nat.dvd_iff_div_mul_eq` | ERROR (step 3) | EXHAUSTED (step 8) | 103 | **1** |

3 theorems escaped the early-error trap and ran to `max_steps` (8) — retrieval *delayed* failure on those theorems but didn't unlock a closer. The 2 remaining ERRORs are theorems where every retrieved tactic at the first failing step also errored.

## Why retrieval did not win any div theorem

Breakdown of the 783 retrieved-tactic outcomes:

| Outcome class | Count |
|---|---|
| `simp made no progress` | 207 |
| `unknown constant` | 200 |
| `type mismatch` | 124 |
| `tactic 'rewrite' failed, equality or iff proof expected` | 108 |
| `tactic 'apply' failed, failed to unify` | 108 |
| `tactic 'rewrite' failed, did not find instance of the pattern` | 30 |
| state-advancing transition | 6 |

Three structural failure modes are visible:

1. **Self-reference trap (200 unknown-constant errors).** 7 distinct lemma names came back as `unknown constant`, including 5 of the **target theorems themselves** (`Nat.div_le_div_right`, `Nat.div_pos`, `Nat.div_pos_iff`, `Nat.div_lt_one_iff`, `Nat.dvd_iff_div_mul_eq`) plus 2 lemmas (`Nat.div_le_iff_le_mul`, `Nat.div_eq_zero_iff`) that exist in Mathlib4 source but are not in the import-closure of the eval environment. The token-overlap retriever happily surfaced the target theorem itself as a top premise on every state, then issued 60+ Lean roundtrips trying `rw [Nat.div_pos]` / `exact Nat.div_pos` to prove `Nat.div_pos` — a pure waste.

2. **Tactic-form misuse (~340 simp/rewrite/apply/exact failures).** The current `{rw, simp, exact, apply}` shotgun fires every tactic form at every premise. Equality-shaped lemmas get `rw`, but propositional lemmas get the same treatment and "no progress" out. Lemma shape (`Iff` vs `Eq` vs `Prop`) is not currently considered.

3. **Premise selection is name-tokenized, not goal-tokenized.** Token overlap ranks `Nat.div_lt_one_iff` ahead of `Nat.mul_div_cancel` on a `Nat.div_lt_one_iff` goal (it shares more name tokens), but only `Nat.mul_div_cancel` and friends are actually rewritable into the goal pattern. The retriever doesn't see the goal *shape* — only the *names*.

These three failure modes are exactly what v4.2 (import-aware tactic availability) and v4.3 (induction-template search) are designed to address; v4.1 surfaces them empirically.

## Verification of seeded lemma names

All lemmas in `STATIC_PREMISES["Nat.div"]` were verified present in `Mathlib/Data/Nat/Defs.lean` (or its transitive imports) at the lean_dojo cache HEAD `29dcec074de168ac2bf835a77ef68bbe069194c5` *before* the run. The 7 that came back `unknown constant` during the Lean rollouts are present in the source file but apparently not in the elaborated import-closure of the `nat_defs_medium` evaluation environment — confirming that "grep in Mathlib4 source" is **not** sufficient verification; a real Lean check would be needed (deferred to v4.2 "import-aware tactic availability").

## Comparison to v3.6

| Metric | v3.6 | v4.1 | Δ |
|---|---|---|---|
| Proved | 25/38 | 25/38 | 0 |
| Errored | 10 | 6 | −4 (4 div theorems moved to EXHAUSTED) |
| Exhausted | 3 | 7 | +4 |
| Avg steps (proved) | 1.1 | 1.1 | 0 |
| Fallbacks used | 24 | 24 | 0 |
| Family activations | `{div:6, mod:5, AM_GM:1}` | `{div:6, mod:5, AM_GM:1}` | identical |
| Family proofs | `{mod:4}` | `{mod:4}` | identical |
| Denied tactics | 8 | 8 | identical |
| Wallclock | ~3m 28s | 5m 22s | +1m 54s |

The Error→Exhausted shift on 4 div theorems is the only behavioral change outside the retrieval layer itself. The 25 wins, their origins, and their winning tactics are bit-identical to v3.6.

## What's plumbed

- `premise_retriever.retrieve_for_state(state_pp, theorem_name, k, family_key)` — stateless, deterministic, returns up to `k` ranked lemma names. v4.1 ships a `Nat.div` bucket with 16 seeded lemmas.
- `StrategyWrapperPolicy(retrieval_enabled=, retrieval_top_k=, retrieval_tactic_forms=)` — when a `div`-family theorem activates, calls the retriever and synthesizes one `rw / simp / exact / apply` per premise. Entries inserted between family and generic, deduped, tagged with `ORIGIN_RETRIEVED = "retrieved_premise"`. Cap automatically bumped by `len(forms) × top_k` when retrieval activates.
- `tactic_retrieved_premise` source field on every retrieved tactic in `traces.jsonl`.
- Aggregate metrics: `retrieved_premise_{activation,attempt,advanced,proved}_count`, `retrieved_premise_wins`.
- Roundtrips through `dump_strategy_config` / `load_strategy_config` so the evaluator subprocess picks up `retrieval_enabled` / `retrieval_top_k` / `retrieval_tactic_forms` from the candidate's JSON.
- `SearchCandidate.retrieval_*` genome fields; `make_seed_candidate("hybrid_evolved", …)` ships `retrieval_enabled=True`, `retrieval_top_k=10`.

## What is not v4.1's job (saved for v4.2 / v4.3)

- **v4.2 — import-aware tactic availability.** Verify a lemma is actually `import`-reachable in the eval env before queuing it. Filter out the target theorem itself. Pick the tactic form by lemma shape.
- **v4.3 — induction-template search.** For `Nat.div_le_div_right`-style theorems whose closure needs `induction n with | zero => … | succ n ih => …` over the div recursion, the wrapper currently has no induction-template variant under the div family.

## Artifacts

- `project/evolve/runs/evolve-20260521-233937-cf2370/` — v4.1 medium run root
- `project/evolve/runs/evolve-20260521-233937-cf2370/eval/seed-baseline/eval-38e7aaf6/metrics.json` — full metrics JSON
- `project/evolve/runs/evolve-20260521-233937-cf2370/eval/seed-baseline/eval-38e7aaf6/traces.jsonl` — 2,094 trace records (783 tagged `retrieved_premise`)
- `project/evolve/runs/evolve-20260521-233937-cf2370/eval/seed-baseline/strategy_config.json` — dumped config used by the eval subprocess (includes `retrieval_enabled=true`, `retrieval_top_k=10`)
- Branch: `v4-premise-retrieval-div` (off `e74861f`)
