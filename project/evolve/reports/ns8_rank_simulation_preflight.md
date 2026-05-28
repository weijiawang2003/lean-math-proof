# NS8 — full ranked-list pre-flight simulation

NS7 (commit `709ec70`) demonstrated that the 20-skeleton compaction
floor is a *rank-coupling barrier*: disabling a credit-zero skeleton
silently shifts the wrapper's merged top-K window so a
*correctly-protected* retrieval skeleton drops out of the ranked list.
NS7's pre-flight detector operated only on the bag's deterministic
skeleton-emit order, missing this second-order effect. NS6 cycle 4
through NS7 cycle 19 all hit the same regression on
`Nat.div_lt_iff_lt_mul'`.

NS8 closes the gap by simulating the wrapper's *full merged ranked
list* — skeleton emissions interleaved with cached `gen_v5` model
outputs — and rejecting any mutation that pushes a protected
*critical tactic* out of that list.

## Hard constraints respected

- No retraining, no checkpoint changes, no broad refactor.
- Preserved: `nat_defs_medium 37/38` and `nat_defs_large_v5 49/65`.
- `use_skeleton_bag` flag unchanged.
- Run artifacts and the per-state model cache under
  `project/evolve/ns8_runs/` and `project/evolve/archive/` (already
  gitignored).
- No LLM mutator; no gen_v5+1 training.

## Stage 1 — protected states

`scripts/ns8_protected_states.py` joins the NS7
`protected_skeletons.json` with per-step traces and emits a
per-state JSONL row carrying:

  - `theorem`, `state_hash`, `full_name`, `step`
  - `state_pp` (verbatim from the trace)
  - `critical_skeleton_stable_id`, `critical_skeleton_name`
  - `critical_tactic` (the actual Lean tactic string)
  - `critical_role` (close / advance)
  - `reason` (direct_win / assist_win / critical_advance)
  - `observed_rank_in_trace`

On the NS7 best (20 enabled skeletons, fresh medium+large traces with
stable IDs): **52 protected states** across 36 distinct theorems.

By reason: `{direct_win: 45, assist_win: 2, critical_advance: 5}`.
By critical-tactic origin: `{tactic_template: 38, family_tactic: 2,
retrieved_premise: 3, fallback_tactic: 9}`.

The 2 assist_win states are exactly the NS5/NS6 must-protects
(`pt_any_13` on `Nat.add_mod_eq_ite`,
`retrieved:Nat.div_lt_iff_lt_mul:rw` on `Nat.div_lt_iff_lt_mul'`).

Output: `project/evolve/archive/protected_states.jsonl`.

## Stage 2 — cached model outputs

`scripts/ns8_cache_model_outputs.py` loads `GenerativePolicy(gen_v5)`
exactly as the eval pipeline does (beam decode, top_k=8, fixed seed),
runs it once per protected state, and persists the top-K tactic
strings keyed by:

```
cache_key = sha1((state_pp, full_name, model_path, decode_mode,
                  top_k, seed))[:16]
```

The cache is resumable (existing rows are detected by key and
skipped) and small (~26 KB for 52 states). Output:
`project/evolve/archive/model_outputs_cache.jsonl`.

## Stage 3 — full ranked-list simulator

The naive approach — re-implement the wrapper's merge in
isolation — would diverge as the wrapper evolves. NS8 instead
instantiates the *real* `StrategyWrapperPolicy` with a
`CachedBasePolicy` that returns cached top-K outputs for any
`(state_pp, full_name)` it has seen:

```
class CachedBasePolicy:
    def rank_tactics(self, state_pp, full_name, k):
        return self._cache[(state_pp, full_name)][:k]
```

The wrapper's existing merge logic does the rest — priority
skeletons, model outputs, family/retrieved/term/fallback, dedup,
cap. The simulator output is byte-equivalent to what the live
eval would produce for that state, given the cached model is
deterministic and the genome is the only varying input.

**Validation on the NS6/NS7 known regression** (cycle 4 mutation:
disable `fb_19, pt_iff_2, fam_div_14`):

| disabled | critical-tactic rank in baseline | rank in mutated |
|---|---:|---:|
| `pt_iff_2` alone | 16 | 15 (forward by 1) |
| `fb_19` alone | 16 | 16 |
| **`fam_div_14` alone** | 16 | **absent** |
| All three (NS6 c4) | 16 | **absent** |

`fam_div_14` is the family activator for the "div" family. The
wrapper's retrieval block only fires when `activated_families` is
non-empty. Disabling the only `family_tactic` skeleton for "div"
means the family doesn't activate, retrieval doesn't fire, and
`retrieved:Nat.div_lt_iff_lt_mul:rw` (the assist skeleton) is never
emitted. NS7's bag-only detector couldn't see this because the
retrieval bag is *dynamic* (synthesized per-state), so the
"skeleton bag" hasn't visibly lost anything.

The simulator catches it exactly.

## Stage 4 — integrated detector

`evolve/rank_coupling.py::check_state_coupling(baseline, mutated,
protected_states, simulator)` walks every protected state, simulates
both genomes' ranked lists, and emits a `StateViolation` whenever the
critical tactic disappears from the mutated list (or is pushed back
past `baseline_rank + slack`).

The NS8 runner (`evolve/skeleton_evolve_ns8.py`) calls this *before*
launching `eval_rollout_all.py`. Pre-flight rejected candidates
never start a Lean subprocess.

## Stage 5 — replay against NS7 regressions

Re-running every NS7 cycle's genome through the NS8 detector:

| NS7 cycle | mutation | NS7 outcome | NS8 detector |
|---|---|---|---|
| 1 | baseline | promoted | SAFE |
| 2 | disable_dead {3,2} | Lean-rej (36) | **REJ (3 thms)** |
| 3 | disable_dead {5,3} | Lean-rej (36) | **REJ (3 thms)** |
| 4 | disable_dead {8,5} | Lean-rej (36) | **REJ (3 thms)** |
| 5 | archive_seed_credit 12 | pre-flight rej | already rejected |
| 6 | archive_seed_credit 15 | Lean-rej (36) | **REJ (3 thms)** |
| 7 | archive_seed_credit 18 | Lean-rej (36) | **REJ (3 thms)** |
| 8 | archive_seed_credit 20 | Lean-rej (36) | **REJ (3 thms)** |
| 9 | archive_seed_credit 22 | Lean-rej (36) | **REJ (3 thms)** |
| 10 | archive_seed_credit 25 | Lean-rej (36) | **REJ (3 thms)** |
| 11-12 | archive_seed wins-only | pre-flight rej | already rejected |
| 13-18 | scoped reorders | accepted | SAFE |
| 19 | disable_dead {5,8} | Lean-rej (36) | **REJ (3 thms)** |
| 20 | disable_dead {10,12} | Lean-rej (36) | **REJ (3 thms)** |
| 21 | baseline | accepted | SAFE |

**10/10 Lean-rejected NS7 cycles are now caught pre-flight.**
At ~150s per Lean medium eval, that's ~25 minutes saved on the same
queue. Scoped-reorder cycles all pass cleanly. Baselines pass.

## Stage 6 — NS8 bounded sweep (20 cycles, ~25 min)

| metric | NS8 |
|---|---:|
| best medium proved   | 37 |
| best large proved    | 49 |
| best enabled skeletons | 20 |
| total cycles         | 20 |
| **pre-flight rejected** | **12** |
| **Lean rejected**       | **0** |
| accepted (no promote) | 7 |
| promoted             | 1 (baseline confirm) |

The 10 prior Lean regressions have all become pre-flight rejections.
Every cycle that *did* run Lean preserved 37. The compaction floor
remains at 20 — the simulator confirms every safe-pruning mutation
*correctly* causes a regression on `Nat.div_lt_iff_lt_mul'` (or
related theorems), and rejects them before paying for Lean.

### Cycle-by-cycle

| c | operator | kwargs | result | preflight |
|---|---|---|---|---|
| 1 | baseline | — | promoted med=37 en=20 | — |
| 2 | disable_dead | {3,2} | PF-REJ 3v | div family |
| 3 | disable_dead | {5,3} | PF-REJ 3v | div family |
| 4 | disable_dead | {8,5} | PF-REJ 3v | div family |
| 5 | archive_seed_credit | 15 | PF-REJ 3v | div family |
| 6 | archive_seed_credit | 18 | PF-REJ 3v | div family |
| 7 | archive_seed_credit | 20 | PF-REJ 3v | div family |
| 8 | archive_seed_credit | 22 | PF-REJ 3v | div family |
| 9 | archive_seed_credit | 25 | PF-REJ 3v | div family |
| 10 | archive_seed (wins-only) | 18 | PF-REJ **9v** | mod/add/div |
| 11 | archive_seed (wins-only) | 22 | PF-REJ **7v** | div_le, div_lt, add_mod |
| 12 | promote_high_win | pri/iff | accepted med=37 | — |
| 13 | promote_high_win | fb/any | accepted med=37 | — |
| 14 | promote_high_win | tt/any | accepted med=37 | — |
| 15 | promote_high_win | fam/any | accepted med=37 | — |
| 16 | demote_generic | pri/iff | accepted med=37 | — |
| 17 | demote_generic | pri/any | accepted med=37 | — |
| 18 | disable_dead | {5,8} | PF-REJ 3v | div family |
| 19 | disable_dead | {10,12} | PF-REJ 3v | div family |
| 20 | baseline | — | accepted med=37 | — |

### Best genome (cycle 1)

Identical to the NS7 best — 20 enabled skeletons preserving 37/49.
The NS8 sweep proves this floor is now *enforced pre-flight*: no
further compaction is possible without breaking `Nat.div_lt_iff_lt_mul'`.

## Examples of simulator output

### Baseline — `Nat.div_lt_iff_lt_mul'` step 1

```
rank=15 family_tactic      fam_div_14                          tac=omega
rank=16 retrieved_premise  retrieved:Nat.div_lt_iff_lt_mul:rw  tac=rw [Nat.div_lt_iff_lt_mul]  <-- CRITICAL
rank=17 retrieved_premise  retrieved:Nat.div_lt_iff_lt_mul:simp tac=simp [Nat.div_lt_iff_lt_mul]
```

### After disabling `fam_div_14`

```
... no family_tactic entries ...
... no retrieved_premise entries ...   <-- retrieval gated on activated_families
rank=15 fallback_tactic    fb_19                               tac=simp_all [Nat.add_mod, Nat.mod_eq_of_lt]
(list length: 17 entries, critical tactic absent)
```

The wrapper's retrieval block at `evolve/strategy_wrapper.py:681` is
gated on `if self.retrieval_enabled and self.retrieval_top_k > 0
and activated_families:`. Disabling the only `family_tactic` for the
`div` family means `activated_families` is empty for theorems
matched solely on `div`, so the retrieval emit is skipped entirely.

## Comparison to NS7

| | NS7 (21 cycles) | NS8 (20 cycles) |
|---|---|---|
| best medium / large | 37/38, 49/65 | 37/38, 49/65 |
| best enabled       | 20 | 20 |
| pre-flight rejections | 3 | **12** |
| Lean rejections    | 10 | **0** |
| Lean minutes saved vs NS6-style runs | ~7 min | **~30 min** |
| detector basis     | bag-only emit order | wrapper-merged ranked list |
| catches family-activation effects | no | **yes** |
| catches retrieval gating effects | no | **yes** |
| catches model-cap effects | partial | **yes** |

## Files added/changed

- `scripts/ns8_protected_states.py` — protected-state extractor
- `scripts/ns8_cache_model_outputs.py` — model-output cacher (resumable)
- `evolve/rank_simulator.py` — `CachedBasePolicy` + `RankSimulator`
- `evolve/rank_coupling.py` — `StateViolation`, `check_state_coupling`,
  `summarize_state_violations`
- `evolve/skeleton_evolve_ns8.py` — NS8 runner with full-simulation pre-flight
- `project/evolve/reports/ns8_rank_simulation_preflight.md` (this file)
- `.gitignore` — `project/evolve/ns8_runs/`,
  `project/evolve/archive/model_outputs_cache.jsonl`,
  `project/evolve/archive/protected_states.jsonl`

## Remaining limitations

1. **Cache invalidation**: the model cache is keyed by state_pp +
   full_name + model_path + decode settings. Changing the model
   (e.g. gen_v5+1) requires re-running `ns8_cache_model_outputs.py`.
   The cache is small (~500 bytes per state) so re-caching is cheap.

2. **New protected states**: when the NS7 best changes (new traces),
   the protected set should be regenerated. The current pipeline
   handles this — `ns8_protected_states.py` + `ns8_cache_model_outputs.py`
   take ~30 s end-to-end for 52 states.

3. **State coverage**: only states observed in baseline traces are
   protected. A mutation that introduces a brand-new path won't be
   evaluated by the detector. The Lean eval is still the final gate.

4. **Per-theorem deny-list interactions**: the wrapper applies
   theorem-specific deny lists after merging. The simulator
   replicates this via the genome's `theorem_tactic_denylist` field,
   so it's accurate as long as the deny list is in the genome (it
   is).

## Recommendation for next step (NS9)

The 20-skeleton floor is now a deterministic, pre-flight-enforced
barrier. To break through, the genome must change in ways the current
operator set doesn't support:

1. **Family-activation as a first-class mutation.** Right now, removing
   any family_tactic skeleton may silently disable retrieval for that
   family. A new operator could *promote* a fallback or
   tactic_template skeleton to family scope (or split a generic into
   per-family clones) to keep retrieval alive without depending on the
   originally-emitted family_tactic skeleton.

2. **Retrieval-gate decoupling.** The wrapper's
   `if activated_families:` precondition for retrieval is the root
   cause. A genome flag (`retrieval_requires_family: bool`) would let
   us emit retrieval candidates even when no family is active.

3. **Per-theorem priority_template injection.** For the single
   surviving regression (`Nat.div_lt_iff_lt_mul'`), the proof step is
   `rw [Nat.div_lt_iff_lt_mul]` then `simp_all`. Adding this as a
   theorem-specific priority_template would prove it without
   retrieval. The genome supports per-theorem-tactic-denylist already;
   per-theorem priority_templates is a symmetric extension.

4. **LLM-suggested mutations.** With the pre-flight detector now
   reliable, an LLM mutator can propose creative skeleton variants
   without burning Lean cycles on bad ones. Pre-flight first; Lean
   only on candidates that survive.
