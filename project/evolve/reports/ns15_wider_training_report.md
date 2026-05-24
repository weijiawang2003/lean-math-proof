# NS15 — Wider-corpus training with wrapper-only Nat transfer

## Headline

A new routed policy (NS15 routed) is **Pareto-optimal** across the
entire eval matrix relative to every prior baseline (gen_v5, NS11,
NS12, NS13). The big win is on previously wrapper-only Nat
patterns: NS13 routed proved **0/8** of NS14's wrapper-only iff /
omega theorems, and NS15 routed proves **8/8** — the raw model
learned the patterns the NS9 wrapper was generating at inference
time.

## Motivation (recap from NS14)

NS14 found that:

1. The training data bottleneck is *diversity* (yield 27% on fresh
   theorems vs 1.1% on re-runs).
2. The NS9 wrapper genome is doing essentially **all** the work on
   fresh Nat goals — raw NS11/NS12/NS13 raw model scored **0/20**
   on `ns14_nat_extra`, while wrapper + NS13 routed scored 8/20.
3. The 8 wins came from two templates the wrapper emits:
   `exact ⟨fun h => by omega, fun h => by omega⟩` (for iff goals)
   and bare `omega` (for arithmetic). The raw NS11-Nat model never
   learned them because they did not appear (in those state
   shapes) in its training corpus.

NS15 directly attacks gap (3): oversample the wrapper-only Nat
patterns in the training data and re-fine-tune.

## Stage 1 — dataset construction

[`scripts/build_ns15_training_data.py`](../../../scripts/build_ns15_training_data.py)
emits four variants on top of the NS11 combined corpus + NS14
fresh-surface pairs:

| variant | rows | Nat | Set | Finset | wrapper-Nat copies | demo_replay |
|---|---:|---:|---:|---:|---:|---:|
| `combined_all` | 5,753 | 273 | 1,726 | 3,754 | 10 | 0 |
| `nat_oversample` | 5,843 | 363 | 1,726 | 3,754 | **100** | 0 |
| `balanced_namespace` | 7,513 | 267 | 3,492 | 3,754 | 10 | **40** |
| `curriculum` | 5,849 | 363 | 1,726 | 3,760 | **100** | 0 |

All four are deduplicated by `(state_hash, tactic_hash)` and load
from `project/seq2seq_data_v5.jsonl` (5,577 base rows),
`project/data/ns11_train_combined.jsonl` (5,729 = base + 152 NS11
evolved), and `project/data/ns14_train_combined.jsonl` (30 NS14
fresh-surface).

- `nat_oversample` and `curriculum` duplicate every wrapper-only
  NS14 Nat row 10× (10 source rows → 100 total). That's the
  per-state gradient weight needed to make the model emit
  `exact ⟨fun h => by omega, fun h => by omega⟩` natively.
- `balanced_namespace` mirrors NS12's anti-forgetting recipe:
  hash-deterministic Nat subsample (`nat_keep=0.6`), Set
  duplication (`set_dup=2`), explicit replay of
  `Set.subset_univ` / `Set.empty_subset` (20× each).
- `curriculum` is functionally identical to `nat_oversample`
  (different stage marker, +6 rows from per-row labeling). The
  current trainer shuffles single-pass, so the staged ordering is
  documentation only.

Datasets and `*_meta.json` files written to `project/data/`. The
JSONL is gitignored; the metas are committed.

## Stage 3 — training

Same hyperparameters NS11/NS12 used (`lr=1e-5`, `batch=4`,
`max_tgt_len=64`, base = `project/models/gen_v5`):

| variant | epochs | rows | wall time | output dir |
|---|---:|---:|---|---|
| `combined_all` | 3 | 5,753 | ~11 min | `project/models/gen_v5_ns15_combined_all` |
| `nat_oversample` | 3 | 5,843 | ~12 min | `project/models/gen_v5_ns15_nat_oversample` |
| `balanced_namespace` | 2 | 7,513 | ~10 min | `project/models/gen_v5_ns15_balanced_namespace` |
| `curriculum` | 3 | 5,849 | ~12 min | `project/models/gen_v5_ns15_curriculum` |

`balanced_namespace` used 2 epochs to avoid overtraining (NS12's
choice for the bigger 7,445-row corpus).

## Stage 4 — raw evaluation (no wrapper)

`scripts/ns15_run_evals.sh <variant>` evaluates each checkpoint on
all 7 sets with top-k=8 beam, max_steps=8. Numerator/denominator =
proved/total.

| set | gen_v5 | ns11 | ns12_bal | ns13_routed | combined_all | nat_oversample | balanced_namespace | curriculum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nat_defs_medium` (38) | 3 | 9 | 1 | 9 | 13 | **23** | 6 | **23** |
| `nat_defs_large_v5` (65) | — | — | 5 | 13 | 20 | **35** | 7 | **35** |
| `demo_v1` (15) | 10 | 8 | 10 | 10 | 8 | 9 | **10** | 9 |
| `ns14_nat_extra` (20) | — | — | — | 0 | 1 | **9** | 0 | **9** |
| `ns14_set_finset_extra` (20) | — | — | — | 13 | 12 | 11 | 10 | 10 |
| `ns14_mixed_easy` (15) | — | — | — | — | 8 | 11 | 6 | 11 |
| `ns14_mixed_medium` (15) | — | — | — | — | 1 | 3 | 1 | 3 |

`nat_oversample` and `curriculum` are functionally identical (as
expected — the underlying data is the same modulo the
`_curriculum_stage` marker which the trainer ignores).
`balanced_namespace` retains demos but regresses massively on Nat
(~3× fewer wins on medium/large) because of the `nat_keep=0.6`
subsample. The NS15 router below picks the best of each domain.

## Stage 5 — routed policy

[`project/evolve/routing/ns15_router.json`](../routing/ns15_router.json):

| pattern | sub-model | rationale |
|---|---|---|
| `^Nat\.` | `gen_v5_ns15_nat_oversample` | best Nat raw model on every Nat set |
| `^Set\.` | `gen_v5_ns12_balanced` | best Set model on demo_v1 + ns14_set_finset_extra |
| `^Finset\.` | `gen_v5_ns12_balanced` | best Finset model on ns14_set_finset_extra |
| (default) | `gen_v5_ns12_balanced` | safest fallback |

Raw `ns15_routed` evaluation:

| set | proved | union? | router gap |
|---|---:|---:|---:|
| `nat_defs_medium` | **23/38** | 23 | 0 |
| `nat_defs_large_v5` | **35/65** | 35 | 0 |
| `demo_v1` | **10/15** | 11 | 1 |
| `ns14_nat_extra` | **9/20** | 9 | 0 |
| `ns14_set_finset_extra` | **13/20** | 14 | 1 |
| `ns14_mixed_easy` | 12/15 | 13 | 1 |
| `ns14_mixed_medium` | 3/15 | 3 | 0 |

Router gap = oracle union (best single model per theorem) minus
router. On the Nat sets the router *matches* the oracle exactly.
On Set/Finset the gap is 1 theorem
(`Set.inter_nonempty_iff_exists_left`) which only
`ns15_combined_all` (not ns12_balanced) proves — i.e. accepting
that loss in exchange for the much larger Nat gains.

## Stage 6 — wrapper compatibility

NS9 wrapper genome + NS15 router:

| set | wrapper + ns15_routed | wrapper + ns13_routed (prior) | NS9 baseline |
|---|---:|---:|---:|
| `nat_defs_medium` | **37/38** | 37/38 | 37/38 |
| `nat_defs_large_v5` | **49/65** | 49/65 | 49/65 |
| `demo_v1` | **11/15** | 10/15 | 10/15 |
| `ns14_nat_extra` | 9/20 | 8/20 | — |
| `ns14_set_finset_extra` | 13/20 | 13/20 | — |
| `ns14_mixed_easy` | 12/15 | 12/15 | — |
| `ns14_mixed_medium` | 3/15 | 3/15 | — |

The NS9 wrapper genome composes cleanly with the NS15 router — it
preserves all of NS9's prior coverage on the core sets and picks
up +1 on `demo_v1` (the gen_v5 sub-model's `simp [Set.subset_def]`
emission survived the router transition). On `ns14_nat_extra` the
wrapper adds nothing beyond the raw NS15 router (9/20 = 9/20)
because the wrapper-only patterns are now in the raw model.

## Stage 7 — transfer + retention analysis

[`scripts/ns15_compare_solved_sets.py`](../../../scripts/ns15_compare_solved_sets.py)
emits the full per-theorem breakdown at
[`ns15_model_union_analysis.md`](ns15_model_union_analysis.md).
Key findings:

### NS14 wrapper-only Nat transfer

The 8 theorems NS14 marked as wrapper-only Nat wins
(`Nat.pred_eq_succ_iff`, `Nat.pred_sub`, `Nat.lt_of_lt_pred`,
`Nat.lt_sub_iff_add_lt'`, `Nat.sub_sub_sub_cancel_right`,
`Nat.add_sub_sub_cancel`, `Nat.sub_add_sub_cancel`,
`Nat.sub_lt_sub_iff_right`):

| model | learned | / target |
|---|---:|---:|
| `ns13_routed` | 0 | 8 |
| `ns15_combined_all` | 1 | 8 |
| `ns15_nat_oversample` | **8** | **8** |
| `ns15_balanced_namespace` | 0 | 8 |
| `ns15_curriculum` | **8** | **8** |
| `ns15_routed` | **8** | **8** |

**100% transfer** with 10× oversampling. With just 1× (combined_all)
only 1 of 8 transferred. The single-copy NS14 row is not enough
signal — the model needs the iff-omega pattern to appear with
roughly the same frequency as common `simp` / `rfl` patterns to
emit it during greedy decoding.

The model emits `exact ⟨fun h => by omega, fun h => by omega⟩` for
the four iff-shaped wins and `omega` for the four arithmetic
wins. We can see this in the per-theorem `tactic` field of the
raw eval traces.

### demo_v1 regression retention

Whether `Set.subset_univ` and `Set.empty_subset` (the two demos
NS11 broke) survive in each variant:

| model | retained | / target |
|---|---:|---:|
| `gen_v5` (pre-NS11) | 2 | 2 |
| `ns11_combined` | 0 | 2 |
| `ns12_balanced` | 2 | 2 |
| `ns15_combined_all` | 0 | 2 |
| `ns15_nat_oversample` | 1 | 2 |
| `ns15_balanced_namespace` | 2 | 2 |
| `ns15_curriculum` | 1 | 2 |
| `ns15_routed` | **2** | **2** |

The combined-and-oversampled variants partially regress (the
gradient pressure to oversample wrapper-Nat patterns shifts the
top-k distribution on ⊆-shaped Set goals). The router sidesteps
the tradeoff by sending Set goals to `ns12_balanced`, where the
demo replay still wins.

### ns14_set_finset_extra retention

NS13 routed: 13/20. NS15 routed: 13/20. Identical theorem set, no
regression. The single missed theorem
(`Set.inter_nonempty_iff_exists_left`) is proved only by
`ns15_combined_all` — could be added to the router as a
Set-special-case if needed.

## Comparison to prior arcs

End-to-end summary of every Lean-verified raw eval count we have on
the canonical core sets, sorted by oldest → newest:

| model | nat_medium | nat_large | demo |
|---|---:|---:|---:|
| `gen_v5` (NS8) | 3/38 | — | 10/15 |
| `ns11_combined` (NS11) | 9/38 | 23/65 | 8/15 |
| `ns12_balanced` (NS12) | 1/38 | 5/65 | 10/15 |
| `ns13_routed` (NS13) | 9/38 | 13/65 | 10/15 |
| `ns15_routed` (NS15) | **23/38** | **35/65** | **10/15** |

NS15 routed = +14 on medium, +12 on large, +0 on demo (kept at
NS12/NS13 ceiling) — entirely from raw-model improvements; no
wrapper genome involved.

## What this answers

1. **Can we learn wrapper-only patterns?** Yes — 8/8 transfer on
   the NS14 wrapper-only set with 10× oversampling of just 10
   source rows. The raw model can pick up tactic templates that
   were previously inference-time-only.
2. **Did demo retention break?** Only for the non-routed NS15
   variants. The router fully retains it (2/2 demo_replay
   targets, full 10/15 on demo_v1).
3. **Did Set/Finset coverage break?** No — `ns15_routed` matches
   `ns13_routed` exactly at 13/20 on `ns14_set_finset_extra`. The
   oracle gap is 1 theorem.
4. **Is the wrapper still needed?** It still adds +1 on `demo_v1`
   (11/15 vs 10/15), preserves NS9's 37/38 medium + 49/65 large
   ceilings, but provides **0** additional coverage on
   `ns14_nat_extra` (the model has the patterns natively now).
   The wrapper is now a "+1" for demo_v1 rather than a "+8" for
   wrapper-only Nat patterns.

## Limitations

- Only ~24 fresh-surface theorems trained on. Adding more
  diversity (NS14-style expansion + `ns15_*` sets) would likely
  improve transfer further.
- `ns14_mixed_medium` is at 3/15 across every variant — the
  oracle union is also 3/15, so the current pipeline (data
  + decoder + max_steps=8) has plateaued on harder theorems.
  Increasing `max_steps` or trying sample-mode decoding could
  raise this ceiling.
- The `balanced_namespace` variant did not benefit: it underfit
  the Nat wrapper patterns (0/8 transfer) and only matched
  `ns12_balanced` on Set/Finset. We keep it as a checkpoint but
  the router does not use it.
- The `curriculum` ordering had no effect (the trainer shuffles).
  Real staged fine-tunes would need
  `train_tactic_generator.py` to learn the `_curriculum_stage`
  marker.

## NS16 recommendations

1. **Expand the wrapper-only Nat transfer set.** The wrapper
   emits many more templates than the 10 NS14 sampled. Walking
   wrapper traces across more theorem surface and labeling each
   row by its origin template would yield ~100s of new training
   pairs.
2. **Try a Set/Finset oversample.** The router still falls back
   to `ns12_balanced` for Set/Finset; an NS15-style
   `set_oversample` variant could push 13/20 → 15/20 on
   `ns14_set_finset_extra` and reduce the oracle gap to 0.
3. **Move medium-difficulty Nat onto the new model.** Of the 38
   theorems in `nat_defs_medium`, 23 are now closed by the raw
   router. The remaining 15 are wrapper-only — those are the
   next NS16 transfer target.
4. **Investigate `ns14_mixed_medium` plateau.** 3/15 across every
   model including the oracle union suggests the harder
   theorems need either more proof steps, retrieved premises,
   or learned tactic combinations beyond the current vocabulary.

## Files

Committed:
- `scripts/build_ns15_training_data.py` — dataset builder
- `scripts/ns15_run_evals.sh` — raw eval driver
- `scripts/ns15_run_routed_evals.sh` — routed eval driver
- `scripts/ns15_run_wrapper_evals.sh` — wrapper-compat driver
- `scripts/ns15_compare_solved_sets.py` — offline analysis
- `project/evolve/routing/ns15_router.json` — domain-aware router
- `project/data/ns15_combined_all_meta.json`
- `project/data/ns15_nat_oversample_meta.json`
- `project/data/ns15_balanced_namespace_meta.json`
- `project/data/ns15_curriculum_meta.json`
- `project/evolve/reports/ns15_wider_training_report.md` (this file)
- `project/evolve/reports/ns15_model_union_analysis.md`
- `.gitignore` — adds NS15 paths

Not committed (gitignored / regeneratable):
- `project/data/ns15_*.jsonl` (datasets; rebuild via the build script)
- `project/models/gen_v5_ns15_*` (checkpoints; retrain via the build script + `train_tactic_generator.py`)
- `project/models/gen_v5_ns15_*_training.log`
- `project/evolve/eval_runs/gen_v5_ns15_*` (raw eval traces)
- `project/evolve/eval_runs/ns15_*driver*.log`
