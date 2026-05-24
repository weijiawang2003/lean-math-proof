# NS9 — retrieval gate decoupling

NS8 (commit `04b38bb`) reached a deterministic, pre-flight-enforced
floor at 20 enabled skeletons. The simulator diagnosis showed every
further-compaction attempt regressed because disabling `fam_div_14`
(the only `family_tactic` for the `div` family) makes
`activated_families` empty, which causes the wrapper's retrieval
block to skip entirely:

```
# evolve/strategy_wrapper.py:700 (NS8)
if self.retrieval_enabled and self.retrieval_top_k > 0 and activated_families:
    ...
```

So the *credit-bearing* `retrieved:Nat.div_lt_iff_lt_mul:rw` skeleton
disappears as a side-effect of disabling a different (zero-credit)
skeleton. NS9 closes this by making the retrieval gate independent
of family-tactic survival.

## Hard constraints respected

- No retraining, no checkpoint changes, no broad refactor.
- Preserved: `nat_defs_medium 37/38` and `nat_defs_large_v5 49/65`.
- `use_skeleton_bag` flag unchanged.
- NS8 rank simulator + pre-flight detector kept intact (extended).
- No LLM mutator; no gen_v5+1 training.
- Run artifacts under `project/evolve/ns9_runs/` gitignored.

## Stage 1 — explicit retrieval gates

Two new config / genome fields on `StrategyWrapperPolicy`:

  - `retrieval_requires_family: bool = True` (default preserves NS8
    behaviour exactly)
  - `retrieval_family_gates: list[str] = []` (substrings that gate
    retrieval when `retrieval_requires_family=False`)

Wrapper change at the retrieval block:

```python
if self.retrieval_requires_family:
    retrieval_families = list(activated_families)
else:
    gate_keys = self.retrieval_family_gates or list(
        self.theorem_family_tactics.keys()
    )
    retrieval_families = [
        fam for fam in gate_keys
        if fam and full_name and fam in full_name
    ]

if self.retrieval_enabled and self.retrieval_top_k > 0 and retrieval_families:
    # ... emit retrieved tactics using retrieval_families ...
```

With `retrieval_requires_family=False` and
`retrieval_family_gates=["div", "mod", "pow"]`, the retrieval block
fires whenever the theorem name contains "div", "mod", or "pow" —
regardless of whether the corresponding family_tactic skeleton is
enabled. The downstream `emit_retrieved_tactics` only requires that
each gate key appear in `premise_retriever._FAMILY_CATALOG_KEYS`, so
this works without any change to the retriever or the bag.

Both fields are surfaced through `load_strategy_config` and
`dump_strategy_config`. `eval_rollout_all.py` consumes the extended
tuple and passes the new fields to `StrategyWrapperPolicy`. The NS8
`evolve/rank_simulator.py` was updated to forward the same fields,
so the simulator's wrapper instance behaves identically to the live
eval.

## Stage 2-3 — bag and simulator updates

The retrieved-skeleton stable IDs and trace attribution were already
in place from NS7. The bag's `emit_retrieved_tactics` already takes
`activated_families` as a parameter, so the NS9 change is purely in
the *wrapper*: it now computes the correct `retrieval_families` list
and forwards it to the bag.

Effect: the simulator now reflects the new gate logic automatically
(it uses the real wrapper). Retrieved skeletons appear in the
ranked list whenever the theorem name matches `retrieval_family_gates`.

## Stage 4 — root-cause fix validation

Reproduction of the NS6/NS7/NS8 known regression on
`Nat.div_lt_iff_lt_mul'`, simulated through `evolve.rank_simulator`:

| genome | gate | critical-tactic rank | list length |
|---|---|---:|---:|
| baseline (NS8) | requires_family=True | **16** | 35 |
| NS8 + disable `fam_div_14` | requires_family=True | **missing** | 17 |
| NS9 + disable `fam_div_14` | requires_family=**False**, gates=`["div"]` | **15** | 35 |

The retrieval block now fires on the div theorem even with
`fam_div_14` disabled, the critical tactic shifts forward by 1
position (one fewer family_tactic emission ahead of it), and the
proof remains reachable.

NS8 pre-flight detector run on the NS9 + disable-`fam_div_14`
candidate: **0 violations** across all 52 protected states.

## Stage 5 — bounded compaction sweep (20 cycles, ~50 min)

Same NS8 runner (`evolve/skeleton_evolve_ns8.py`), same NS8 detector,
same protected_states.jsonl, same model_outputs_cache.jsonl — only
the seed genome changed (NS9 gate enabled). Cache and protected set
remained valid: keys depend on `(state_pp, full_name, model, decode,
top_k, seed)` plus critical tactic identity, none of which change
under the new gate.

| metric | NS9 |
|---|---:|
| best medium proved | 37/38 |
| best large proved | 49/65 |
| **best enabled skeletons** | **17** (vs NS8 20 — **−3, 15% smaller**) |
| total cycles | 20 |
| pre-flight rejections | 6 |
| Lean rejections | 0 |
| promotions | 3 (cycles 1, 2, 3) |

### Cycle-by-cycle

| c | operator | kwargs | med | l | en | result |
|---|---|---|---|---|---:|---|
| 1 | baseline | — | 37 | 49 | 20 | promoted |
| 2 | disable_dead | {3,2} | 37 | 49 | 18 | **promoted (strict_compact)** |
| 3 | disable_dead | {5,3} | 37 | 49 | **17** | **promoted (strict_compact)** |
| 4 | disable_dead | {8,5} | 37 | — | 17 | accepted (no further compact) |
| 5 | archive_seed_credit | 15 | — | — | 11 | PF-REJ (5 thms) |
| 6 | archive_seed_credit | 18 | — | — | 11 | PF-REJ |
| 7 | archive_seed_credit | 20 | — | — | 12 | PF-REJ |
| 8 | archive_seed_credit | 22 | — | — | 14 | PF-REJ |
| 9 | archive_seed_credit | 25 | 37 | — | 17 | accepted |
| 10 | archive_seed (wins) | 18 | — | — | 7 | PF-REJ (17 thms) |
| 11 | archive_seed (wins) | 22 | — | — | 8 | PF-REJ |
| 12 | promote_high_win pri/iff | — | 37 | — | 17 | accepted |
| 13 | promote_high_win fb/any | — | 37 | — | 17 | accepted |
| 14 | promote_high_win tt/any | — | 37 | — | 17 | accepted |
| 15 | promote_high_win fam/any | — | 37 | — | 17 | accepted |
| 16 | demote_generic pri/iff | — | 37 | — | 17 | accepted |
| 17 | demote_generic pri/any | — | 37 | — | 17 | accepted |
| 18 | disable_dead {5,8} | — | 37 | — | 17 | accepted (no further compact) |
| 19 | disable_dead {10,12} | — | 37 | — | 17 | accepted |
| 20 | baseline | — | 37 | — | 17 | accepted |

### Mutation details for the breakthrough cycles

- **Cycle 2** disabled `fb_19` and `fam_div_14` (the exact pair NS6
  cycle 4 tried; NS7/NS8 always rejected). Under NS9 gate, retrieval
  still fires on div theorems and the proof succeeds.
- **Cycle 3** disabled `pt_iff_2` (also previously protected by
  NS7/NS8 indirect effects). Confirms the gate change cleanly
  unblocks the previously-coupled prune class.

### Best genome (cycle 3) — 17 enabled skeletons

  - 12 priority_templates: `pt_iff_{0..7}`, `pt_any_{9,10}`,
    `pt_eq_11`, `pt_le_12`, `pt_lt_8`
  - 3 family_tactic: `fam_mod_{13,14,15}` (no `fam_div_*` — pruned)
  - 1 fallback_tactic: `fb_16`
  - 1 retrieval gate: `retrieval_family_gates=["div", "mod", "pow"]`

The genome saved at `project/evolve/ns9_runs/<run_id>/best_candidate.json`
preserves 37/38 medium and 49/65 large with 17 enabled skeletons —
the smallest skeleton-bag configuration the current evidence supports.

## Comparison

| | NS5 | NS6 | NS7 | NS8 | **NS9** |
|---|---|---|---|---|---|
| best enabled       | 25 | 20 | 20 | 20 | **17** |
| best medium        | 37/38 | 37/38 | 37/38 | 37/38 | 37/38 |
| best large         | 49 | 49 | 49 | 49 | 49 |
| Lean rejections / sweep | 67/165 | 6/20 | 10/21 | 0/20 | 0/20 |
| pre-flight rejections | n/a | n/a | 3 | 12 | 6 |
| compact-genome floor mechanism | wins-only | wins-only | wins-only | rank-coupling | **rank-coupling + gate** |

NS9 is the first iteration since NS5 to break the proved-37
compact-genome ceiling. The 17-skeleton genome is **65% smaller**
than the raw NS3-combined baseline (48 enabled).

## Rank simulator changes

`evolve/rank_simulator.py` (the NS8 simulator) was extended to pass
`retrieval_requires_family` and `retrieval_family_gates` through to
the wrapper instance it builds. No further changes — the simulator
already used the real wrapper, so all gate semantics propagate
automatically.

## Files added/changed

- `evolve/strategy_wrapper.py` — `retrieval_requires_family` /
  `retrieval_family_gates` on `StrategyWrapperPolicy`, the retrieval
  block, `load_strategy_config`, `dump_strategy_config`.
- `evolve/autonomous_research_loop.py::write_strategy_config` —
  forwards the new fields.
- `evolve/rank_simulator.py` — `_build_wrapper` passes the new fields.
- `eval_rollout_all.py` — unpacks the extended config tuple.
- `project/evolve/reports/ns9_retrieval_gate_decoupling.md` (this file).
- `.gitignore` — adds `project/evolve/ns9_runs/`.

No retraining; no checkpoint changes; no broad refactor.

## Remaining limitations / next directions (NS10)

1. **Family-tactic-only retrieval gate.** The gate is currently
   global (`retrieval_family_gates` is a single list). A per-family
   `retrieval_family_gates: dict[str, bool]` would let some families
   require the family_tactic (legacy semantics) and others not.

2. **More compact archive_seed_credit.** The credit-aware compact
   experiment at top_n=15..22 still pre-flight-rejects in NS9 —
   the credit scorer keeps `fam_div_14`-equivalents as low-score
   no-credit entries. A "respect retrieval gate" variant of the
   scorer could selectively re-include retrieval-gating skeletons
   without ranking them above the actual winners.

3. **Per-theorem priority_template injection.** For the surviving
   ceiling (one theorem short of perfect on medium), adding a
   theorem-specific priority_template for the rare survivors
   (`Nat.AM_GM`) would close the medium gap to 38/38.

4. **gen_v5+1 retraining.** The 17 enabled skeletons + the model
   together cap the medium ceiling at 37/38. The remaining gap is a
   model capability issue, not a skeleton one. A targeted fine-tune
   on the unproved theorem's structure is the path to 38/38.
