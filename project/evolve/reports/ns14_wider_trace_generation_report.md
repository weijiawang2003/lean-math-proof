# NS14 — Wider-surface trace generation

## Motivation

NS11 walked all 368 existing trace files (13,773 episodes) and
extracted only **152** unique (state, tactic) supervised pairs.
NS12 confirmed the bottleneck is **diversity, not depth**: the
wrapper kept re-hitting the same 51 theorems with the same
tactics regardless of how many cycles we ran.

NS13's domain router gave us a Pareto-optimal *inference* policy
but produced no new training data. NS14 attacks the data
bottleneck directly: enumerate fresh theorems, run the strongest
available policy (NS9 wrapper + NS13 routed base) over them, and
keep every Lean-verified transition.

## Stage 1 — wider theorem-set construction

[`scripts/build_ns14_theorem_sets.py`](../../../scripts/build_ns14_theorem_sets.py)
reads `project/discovered_theorems.json` (527 theorems across
3 confirmed-available Mathlib files), filters out everything
already in our existing eval sets, and emits four new sets to
`project/evolve/routing/ns14_theorem_sets.json`. The patched
[`tasks.py`](../../../tasks.py) loads them at import time so
they're addressable from `eval_rollout_all.py`.

| set | size | by namespace |
|---|---:|---|
| `ns14_nat_extra` | 20 | Nat 20 |
| `ns14_set_finset_extra` | 20 | Set 10, Finset 10 |
| `ns14_mixed_easy` | 15 | Nat 5, Set 5, Finset 5 |
| `ns14_mixed_medium` | 15 | Nat 5, Set 5, Finset 5 |

Total fresh surface: **70 theorems** (some overlap between the
two mixed sets and the per-namespace sets). All 70 are not
previously evaluated.

## Stage 2 — policy runs (top-k=8 beam, max_steps=8)

| policy | set | proved |
|---|---|---:|
| NS9 wrapper + routed | ns14_set_finset_extra | **13/20** |
| NS9 wrapper + routed | ns14_nat_extra | 8/20 |
| NS9 wrapper + routed | ns14_mixed_easy | 12/15 |
| NS9 wrapper + routed | ns14_mixed_medium | 3/15 |
| raw routed | ns14_set_finset_extra | 13/20 |
| raw routed | ns14_nat_extra | **0/20** |

Two findings stand out:

1. **Raw routed matches wrapper on Set/Finset (13/20 vs 13/20).**
   Every new Set/Finset win is already in the raw model's top-k.
   The wrapper adds nothing here — its NS9 genome was tuned for
   the Nat-Defs medium set, and Set/Finset closure is driven by
   the base model's `aesop` / `simp [Set.subset_def]` /
   `simp [Set.ext_iff]` emissions.
2. **Raw routed gets 0/20 on ns14_nat_extra.** All 8 wrapper wins
   come from the wrapper's priority-template + family-tactic
   emissions (e.g. `exact ⟨fun h => by omega, fun h => by omega⟩`
   for iff goals; `omega` for arithmetic). The raw NS11-Nat
   checkpoint did not learn to emit these patterns on the *new*
   goal shapes, only on the goal shapes it saw in training. Net:
   **the wrapper still provides ~100% of Nat coverage on unseen
   theorems** — a clear signal for NS15.

### New theorems closed (deduplicated)

Set/Finset (13 new):
- `Set.ext_iff`, `Set.forall_in_swap`, `Set.not_subset`,
  `Set.not_nonempty_iff_eq_empty'`, `Set.isEmpty_coe_sort`,
  `Set.ne_univ_iff_exists_not_mem`,
  `Set.not_subset_iff_exists_mem_not_mem`,
  `Finset.forall_mem_not_eq`, `Finset.forall_mem_not_eq'`,
  `Finset.coe_ssubset`, `Finset.not_mem_empty`,
  `Finset.coe_empty`, `Finset.isEmpty_coe_sort`.

Nat extra (8 new, wrapper-only):
- `Nat.pred_eq_succ_iff`, `Nat.pred_sub`, `Nat.lt_of_lt_pred`,
  `Nat.lt_sub_iff_add_lt'`, `Nat.sub_sub_sub_cancel_right`,
  `Nat.add_sub_sub_cancel`, `Nat.sub_add_sub_cancel`,
  `Nat.sub_lt_sub_iff_right`.

Mixed_medium adds (3 new):
- `Nat.one_lt_iff_ne_zero_and_ne_one`, `Nat.two_le_iff`,
  `Finset.singleton_inter_of_not_mem`.

## Stages 3 + 4 — pair extraction with quality filters

[`scripts/build_ns14_training_data.py`](../../../scripts/build_ns14_training_data.py)
walks all `ns14_*` traces.jsonl files, classifies each transition
as `close` or `advance_assist` (downstream-close within K=3
accepted steps), and applies the NS11 filter pipeline:

- no theorem self-reference,
- tactic length ≤ 200, state length ≤ 2500,
- no `LeanError`, no `SkippedBloatingApply`, no
  `SkippedKnownError`, no `loop_detected` / `bloat_rejected`,
- origin must be on the allow-list (NS11 set + retrieved_premise +
  skeleton_emitted),
- dedup by `(state_hash, tactic_hash)`,
- namespace + role + origin labels propagated for downstream
  curricular weighting.

For rows where the raw rollout had no `tactic_origin` field, an
origin tag is inferred from the eval-run directory name
(``ns14_routed_raw_*`` → ``raw_routed``, ``ns14_routed_wrapper_*``
→ ``wrapper_routed``, etc.).

### Yield

| metric | value |
|---|---:|
| trace files scanned | 6 |
| rows pre-dedup | 61 |
| rows post-dedup | **30** |
| unique theorems | 24 |
| close transitions | 24 |
| advance_assist transitions | 6 |

### Distributions

By namespace (post-dedup):

| Nat | Set | Finset |
|---:|---:|---:|
| 12 | 10 | 8 |

By origin:

| origin | rows |
|---|---:|
| raw_routed | 17 |
| fallback_tactic | 6 |
| tactic_template | 4 |
| generative_topk | 3 |

By role: close 24, advance_assist 6.

By source run:

| run | rows (pre-dedup) |
|---|---:|
| ns14_routed_wrapper_set_finset | 17 |
| ns14_routed_wrapper_mixed_easy | 14 |
| ns14_routed_wrapper_nat | 9 |
| ns14_routed_wrapper_mixed_medium | 4 |
| ns14_routed_raw_set_finset | 17 |
| ns14_routed_raw_nat | 0 |

## Comparison to the NS11/NS12 152-pair bottleneck

NS11 yield: **152 unique pairs / 13,773 episodes ≈ 1.1% yield**.
That's because every wrapper run hit the *same* 51 theorems with
the *same* tactics; dedup pruned ~99% of the rows away.

NS14 yield: **30 unique pairs / ~110 episodes ≈ 27% yield**. The
yield went up ~24× because every run touches new theorems with
new states.

Net training-corpus growth potential:

| dataset | unique pairs | unique theorems | yield per episode |
|---|---:|---:|---:|
| NS11 evolved (medium variant) | 152 | 51 | ~1.1% |
| NS14 evolved (this stage) | 30 | 24 | ~27% |
| NS11 + NS14 combined | ~180 | ~75 | — |

The headline result: **24 of the 30 new pairs come from
theorems that were previously unobserved**. None of them appear in
the original 152.

## What this answers

1. **Is there a data-bottleneck cure?** Yes — running over fresh
   theorem surface produces ~24× higher yield per episode than
   re-running on the same surface.
2. **Where does the new signal come from?** Set/Finset domain
   yields 18 of 30 pairs from raw + wrapper combined.
   Nat extras yield 12 pairs (largely wrapper-only).
3. **Is the wrapper still load-bearing for Nat?** Yes — raw
   routed proves 0/20 on ns14_nat_extra; all 8 wrapper wins are
   genome-driven (priority template + omega family).
4. **Could we generate 1000s of new pairs this way?** Plausibly.
   The Mathlib scan saw 527 theorems across just 3 files. We
   exercised only 70 (13%) of them in this stage; expanding to
   the remaining ~457 would multiply the yield. Other Mathlib
   files (List, Option, Multiset, Bool) are listed in the scan
   but with much smaller counts — adding 1 or 2 more files could
   reach ~700 fresh theorems easily.

## Limitations

- Only one decoding regime was tried (top-8 beam at temp 0.8).
  Sampling with temperature > 1.0 or larger top-k could find more
  per-theorem proofs but at higher Lean evaluation cost.
- Both raw and wrapper variants reused the same routed base
  (NS13). We did **not** re-run with `gen_v5` directly because
  every prior eval suggests it adds nothing the router doesn't.
- No retrieved-premise or skeleton-emitted rows showed up — the
  current wrapper genome does not activate those origins on these
  theorem shapes. That's expected; it does not invalidate the
  pipeline.
- The `--no-overlap` policy is enforced only via the existing-set
  filter in Stage 1; if a new Mathlib commit adds more theorems
  the set should be regenerated.

## NS15 recommendations

1. **Train a curriculum-aware corpus.** Merge the 30 NS14 pairs
   with the 152 NS11 pairs and the 5,577 v5 base, with explicit
   namespace balancing (the NS12 lesson: ~30%/30%/30% Nat/Set/
   Finset; never let any single domain dominate the gradient).
   Re-evaluate on demo_v1 to confirm no regression.
2. **Train the model to emit the wrapper's omega/iff patterns
   natively.** All 8 NS14 Nat wins come from wrapper genome,
   which means the model has *never seen* `exact ⟨fun h => by
   omega, fun h => by omega⟩` for these specific state shapes.
   Adding them to the training set should let the raw model close
   these natively in NS15+.
3. **Grow the theorem surface.** Add ns15_* sets pulling from the
   remaining ~457 unused discovered theorems. Lean evaluation is
   cheap relative to training; this is the highest-yield lever.
4. **Try sample-mode decoding (temperature 0.9–1.0)** on the
   already-proved theorems to harvest tactical diversity — at
   modest Lean cost, this could 2–3× the pair count per theorem.

## Files

Committed:
- `scripts/build_ns14_theorem_sets.py` — theorem-set construction
- `scripts/build_ns14_training_data.py` — pair extraction
- `project/evolve/routing/ns14_theorem_sets.json` — chosen 70 theorems
- `project/data/ns14_train_combined_meta.json` — yield metadata
- `tasks.py` — patched to load NS14 sets at import
- `.gitignore` — adds `project/data/ns14_*.jsonl` exclusion
- `project/evolve/reports/ns14_wider_trace_generation_report.md` (this file)

Not committed (gitignored / regeneratable):
- `project/data/ns14_train_combined.jsonl` (30 rows, ~9 KB; rebuild via the scripts above)
- `project/evolve/eval_runs/ns14_*` (raw eval traces; regenerate via Stage 2 invocations above)
- `project/evolve/eval_runs/ns14_*.log`
