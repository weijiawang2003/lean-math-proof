# NS17 — Pattern-family mining + homogeneous pool audit

## Headline

**NS18 training is not justified by the data.** On every fresh
theorem surface we probed (114 new theorems across hard Nat /
Set / Finset / List / Multiset), the NS9 wrapper added **zero**
wins beyond raw NS15 routed. The pattern-family audit confirms
the same picture from the row side: zero families satisfy the
NS18 gate (≥ 10 wrapper-only rows or ≥ 20 pool size AND
≥ 20 unique theorems AND consistent tactic surface).

The wrapper still adds value on the **older Nat surfaces**
(`nat_defs_medium` +14, `nat_defs_large_v5` +14, `demo_v1` +1)
because those sets contain NS11-held-out theorems and shapes the
model never trained on. But on **fresh easy theorem surface**,
wrapper-only signal has converged to zero.

## Stage 1+2+3 — pattern-family audit

[`scripts/ns17_pattern_family_audit.py`](../../../scripts/ns17_pattern_family_audit.py)
inventories every supervised pair we have, partitioned into
``v5_base`` (5,577 rows, 1 theorem name — the original synthetic
seq2seq dataset), ``evolved`` (201 rows / 86 theorems — the
union of NS11 / NS14 / NS16 wrapper-derived training pairs), and
``traces_close_only`` (90 deduplicated closing rows mined from
existing wrapper trace JSONLs not yet absorbed). It greedy-
classifies each tactic into one of ~20 families.

Full output: [`ns17_pattern_family_audit.md`](ns17_pattern_family_audit.md)
and `project/data/ns17_family_audit.json`.

Top families in the evolved partition:

| family | rows | thms | wrapper-only | held-out | example |
|---|---:|---:|---:|---:|---|
| `fallback_omega` | 58 | 32 | 0 | 0 | `omega` |
| `other` | 49 | 35 | 9 | 7 | `by_cases hc : c = 0 <;> [simp [hc]; …]` |
| `iff_omega_pair` | 28 | 27 | 0 | 0 | `exact ⟨fun h => by omega, fun h => by omega⟩` |
| `constructor_omega` | 16 | 16 | 0 | 0 | `constructor <;> intro h_split <;> omega` |
| `nat_simp_arith` | 14 | 11 | 4 | 0 | `simp [Nat.add_mod, Nat.mod_eq_of_lt]` |
| `fallback_aesop` | 13 | 13 | 0 | 0 | `aesop` |
| `set_subset_simp` | 5 | 5 | 0 | 0 | `simp [Set.subset_def]` |
| `set_ext_simp` | 5 | 5 | 0 | 0 | `simp [Set.ext_iff]` |

Key observations:

1. **`iff_omega_pair` is bigger than NS15 used.** NS15 oversampled
   4 NS14 rows; the full corpus now contains 28 distinct
   (state, tactic) pairs across 27 different theorems — and all
   28 are flagged as zero wrapper-only because they were either
   raw-passing or come from NS11/NS14 (which never carried a
   `wrapper_only` flag). The NS15 model already learned this
   pattern.
2. **`constructor_omega` is a NEW family I had not isolated
   before.** 16 rows / 16 theorems with `constructor <;> intro
   h_split <;> omega` (or close variant). The model may not yet
   emit this top-k natively even though we have the data — but
   the wrapper-only count is 0, which means every theorem where
   `constructor_omega` closes is *also* closed by raw on some
   other tactic. So the family is redundant for transfer.
3. **`nat_simp_arith` has 4 wrapper-only rows across 4
   theorems.** The smallest family with non-zero wrapper-only
   signal that is *also* homogeneous.

## Stage 4+5 — fresh-surface probe (where do wrapper templates still fire?)

[`scripts/build_ns17_theorem_sets.py`](../../../scripts/build_ns17_theorem_sets.py)
emits four new theorem sets totaling 114 fresh theorems:

| set | size | namespace |
|---|---:|---|
| `ns17_nat_remaining` | 31 | Nat (all "hard" — the remaining 31 unused Nat lemmas) |
| `ns17_set_extra` | 30 | Set easy |
| `ns17_finset_extra` | 30 | Finset easy |
| `ns17_list_multiset` | 23 | List 13 + Multiset 10 (entirely unexplored namespaces!) |

Per-surface evaluation:

| set | raw NS15 routed | wrapper + NS15 routed | wrapper-only |
|---|---:|---:|---:|
| `ns17_nat_remaining` (31, hard) | 1 | 1 | **0** |
| `ns17_set_extra` (30, easy) | 18 | 18 | **0** |
| `ns17_finset_extra` (30, easy) | 12 | 12 | **0** |
| `ns17_list_multiset` (23) | 11 | 11 | **0** |
| **TOTAL** | **42/114** | **42/114** | **0** |

**Zero wrapper-only wins** on any surface. The NS9 wrapper genome
contributes 0 fresh wins on Set / Finset / List / Multiset / hard
Nat — every theorem the wrapper proves on these sets is also
proved by raw NS15 routed.

Raw NS15 routed performance is actually solid here — 18/30 on
fresh Set, 12/30 on fresh Finset, 11/23 on List/Multiset (the
first time we've ever tested it on List/Multiset). This is more
than 50% yield on Set, ~40% on Finset, ~48% on List/Multiset, all
without wrapper help.

## Stage 6 — family pool aggregation

[`scripts/ns17_build_family_pools.py`](../../../scripts/ns17_build_family_pools.py)
computes the combined pool size (evolved + trace) and the
NS18 readiness gate.

| family | evolved rows | evolved thms | wrapper-only | trace rows | pool | oversample to 100 | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `fallback_omega` | 58 | 32 | 0 | 19 | 77 | 2× | PASS |
| `other` | 49 | 35 | 9 | 10 | 59 | 2× | PASS |
| `iff_omega_pair` | 28 | 27 | 0 | 28 | 56 | 2× | PASS |
| `fallback_aesop` | 13 | 13 | 0 | 11 | 24 | 5× | PASS |
| `nat_simp_arith` | 14 | 11 | 4 | 5 | 19 | 6× | fail |
| `constructor_omega` | 16 | 16 | 0 | 0 | 16 | 7× | fail |
| `set_subset_simp` | 5 | 5 | 0 | 6 | 11 | 10× | fail |
| `set_ext_simp` | 5 | 5 | 0 | 5 | 10 | 10× | fail |
| `exact_named` | 3 | 2 | 2 | 2 | 5 | 20× | fail |
| `simp_baseline` | 4 | 4 | 0 | 1 | 5 | 20× | fail |
| `nat_div_rw` | 2 | 2 | 2 | 2 | 4 | 25× | fail |
| `split_ifs_omega` | 2 | 1 | 1 | 1 | 3 | 34× | fail |

Families that *do* pass the gate by row count
(`fallback_omega`, `other`, `iff_omega_pair`, `fallback_aesop`)
have **zero wrapper-only signal** — the model already emits
these. Training on more copies would just be memorization on
already-solved theorems.

Families that *do* have wrapper-only signal
(`exact_named` 2 WO, `nat_div_rw` 2 WO, `split_ifs_omega` 1 WO,
`apply_named` 1 WO, `nat_simp_arith` 4 WO) all have pools below
20 and most below 5 — well under the threshold for transfer.

## Stage 7 — decision gate

**No family meets all three NS18 readiness criteria:**

1. *Sufficient row count:* ≥ 10 wrapper-only rows OR pool ≥ 20.
2. *Consistent tactic surface:* small example-tactic count
   (homogeneous template).
3. *Held-out sibling surface available:* fresh theorems likely
   to use this family.

The closest candidates are:
- `nat_simp_arith` (4 WO, 19 pool, 6× to reach 100): partially
  homogeneous (simp variants on Nat arithmetic lemmas) but at
  4 wrapper-only rows the NS15 lesson predicts no transfer.
- `set_subset_simp` (0 WO, 11 pool, 10× to reach 100): the model
  already emits this (10/15 on demo_v1).

**Conclusion: do not run NS18 training on the current corpus.**
The hypothesis that transfer requires homogeneous per-family
density is preserved; the data does not provide a pool meeting
that bar.

## Stage 8 — wrapper-expansion probe (recommendations only, not run)

The user's instruction was to make this stage small. Rather than
modifying the live NS9 genome, this report enumerates *which*
wrapper additions would plausibly unlock new training data, to
be evaluated in a future stage (NS18 or NS19) as a focused
experiment.

Candidate wrapper additions:

1. **`aesop` as controlled fallback on Nat.** Currently the
   wrapper does not invoke `aesop` on Nat goals. The audit shows
   13 evolved + 11 trace `fallback_aesop` rows across Set/Finset
   but ~0 on Nat. Adding it for specific Nat shapes
   (linear_arithmetic, divisibility) may unlock new
   wrapper-only wins.
2. **`constructor <;> omega` template for iff goals where the
   simpler `exact ⟨fun h => by omega, fun h => by omega⟩` fails.**
   The audit found a 16-row `constructor_omega` family already in
   the wrapper output; surfacing it as a top-priority emission
   may close more iff goals than the simple template.
3. **`simp [Nat.add_mod, Nat.mul_mod, Nat.mod_eq_of_lt]` family.**
   The `nat_simp_arith` family has 4 wrapper-only wins already;
   a priority template that bundles these specific simp args
   could systematically close the remaining ns16_nat_div_mod
   / ns16_nat_mixed surface (currently 0/25 and 0/28 raw).
4. **`split_ifs <;> omega` for if-then-else Nat goals.** Only 1
   wrapper-only row in evidence, but the template is unique and
   may apply broadly to e.g. `Nat.div`, `Nat.mod`, conditional
   recursion.
5. **Add `decide` and `rfl` retries on Bool/Option/Multiset
   goals.** We have 0 rows of supervision on these namespaces,
   and the NS17 probe shows the raw model is already at
   11/23 on List/Multiset. A wrapper that probes `decide` /
   `simp_all` could push that higher and produce trainable rows
   for a future namespace expansion.

Each of these would need a small genome patch + a wrapper-only
re-eval to confirm new wrapper-only wins. The patch + eval cycle
is on the order of 30 minutes; the NS18 question is whether to
run that 5 ways and harvest the highest-yield template family,
or to abandon wrapper-only training in favor of a different
training signal entirely (DPO, KL-to-base, sample-mode
data augmentation).

## What this answers

1. **Where did NS15's transfer come from?** The 8/8 NS14
   wrapper-only Nat wins came from a single homogeneous template
   (`exact ⟨fun h => by omega, fun h => by omega⟩`) hitting 10
   rows in the training set at 10×. The audit confirms the family
   now has 28 distinct evolved rows across 27 theorems — well
   above the NS15 threshold.
2. **Why didn't NS16 transfer?** The 19-row NS16 corpus mixed 4
   different families (`tactic_template`, `family_tactic`,
   `generative_topk`, `retrieved_premise`) with average ~5 rows
   per family. No family hit the transfer threshold.
3. **Is the wrapper still useful?** Yes for the *old* sets
   (medium/large held-out), zero on *fresh* surface. The wrapper
   is now a held-out-coverage tool, not a general-purpose
   capability gap-filler.
4. **What pool sizes do we have?** Four families pass the basic
   pool-size gate (omega, "other", iff-omega-pair, aesop) but
   none have wrapper-only signal. Five families have wrapper-only
   signal but tiny pools.
5. **Where should NS18 go?** Either grow wrapper templates first
   (Stage 8 candidates) or switch to a different training signal.
   The supervised-fine-tune-on-wrapper-only-rows recipe has hit
   diminishing returns.

## Limitations

- The trace partition only mined the *closing* transitions from
  existing wrapper traces. Advance-assist transitions could
  multiply rows 2-3×, especially for multi-step wrapper proofs.
- The family classifier is heuristic. The 49-row "other" bucket
  contains varied tactics; some might be a homogeneous family
  the regex missed.
- We did not run wrapper on Bool/Option/Sym/etc — those
  namespaces remain unexplored and could yield new wrapper-only
  patterns.
- The wrapper-only count for NS11/NS14 rows is 0 only because
  those datasets predate the `wrapper_only` flag. Re-checking
  per-theorem against raw-eval-results would refine this.

## NS18 recommendations

If NS18 still aims at SFT-on-wrapper-only-rows, **add Stage 8
wrapper expansion as Stage 1 of NS18**, then re-mine. Otherwise:

1. **Switch training signal.** Try preference-based training
   (DPO) on (wrapper-success, raw-failure) pairs rather than
   pure SFT. This uses the wrapper-only data more efficiently
   per row.
2. **Mine assist transitions, not just closes.** The audit
   counted close rows almost exclusively. Multi-step wrapper
   proofs have intermediate states the model never trained on.
3. **Move to harder theorems** (`ns17_nat_remaining` only got
   1/31 raw). These need either richer search or a stronger
   wrapper.
4. **Cross-domain probes.** The 11/23 List/Multiset baseline
   surprised us — the model generalizes domain-free patterns
   (`rfl`, `simp`, `aesop`) better than we assumed. Running a
   wider Bool/Option/Sym probe might find new transfer
   opportunities.

## Files

Committed:
- `scripts/ns17_pattern_family_audit.py` — family inventory + report
- `scripts/build_ns17_theorem_sets.py` — fresh surface construction
- `scripts/ns17_run_evals.sh` — raw vs wrapper probe driver
- `scripts/ns17_build_family_pools.py` — pool aggregation + gate
- `tasks.py` — `_load_ns17_sets()` registers ns17_* sets at import
- `project/evolve/routing/ns17_theorem_sets.json` — 114 fresh theorems
- `project/data/ns17_family_audit.json` — per-family inventory
- `project/data/ns17_family_pools_meta.json` — pool + gate decision
- `project/evolve/reports/ns17_pattern_family_audit.md`
- `project/evolve/reports/ns17_pattern_family_mining_report.md` (this file)
- `.gitignore` — NS17 paths

Not committed (gitignored / regeneratable):
- `project/evolve/eval_runs/ns17_*`
