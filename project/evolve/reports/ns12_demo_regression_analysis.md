# NS12 Stage 1 — demo_v1 regression diagnosis

## Headline

`gen_v5` proves 10/15 on `demo_v1`. After NS11 fine-tuning on the
combined 5,729-pair corpus (v5 base + 152 evolved Nat-heavy pairs),
`gen_v5_ns11_combined` proves 8/15. Same loss for
`gen_v5_ns11_medium` and `gen_v5_ns11_coverage`.

## Lost theorems (all three NS11 variants)

| theorem | gen_v5 winning tactic | ns11_combined top-8 emissions | new outcome |
|---|---|---|---|
| `Set.subset_univ` | `simp [Set.subset_def]` | `simp [Nat.mul_one]`, `simp [Nat.sub_self]`, `simp [List.length_cons]`, `simp [Nat.one_mul]`, `simp [Set.ext_iff]`, `simp [List.map]`, `simp [List.filter]`, `simp [*]` | "All top-8 errored at step 1" |
| `Set.empty_subset` | `simp [Set.subset_def]` | `simp [Nat.mul_one]`, `simp [Set.ext_iff]`, `simp [Nat.sub_self]`, `simp [List.length_cons]`, `simp [List.map]`, `simp [List.filter]`, `simp [*]`, `simp_all` | "All top-8 errored at step 1" |

`simp [Set.subset_def]` is **completely absent** from top-8 for these
two goal shapes (`⊢ s ⊆ univ` and `⊢ ∅ ⊆ s`). The Nat-heavy fine-tune
pushed Nat/List `simp` arguments to the head of the distribution.

## Verification: `simp [Set.subset_def]` is not deleted from vocab

`gen_v5_ns11_combined` still emits `simp [Set.subset_def]` for the
*other* two Set ⊆-shaped goals it sees:

| theorem | ns11_combined winning tactic |
|---|---|
| `Set.inter_univ` | `simp [Set.subset_def]` |
| `Set.univ_inter` | `simp [Set.subset_def]` |

So the tactic exists in the vocabulary; it just dropped out of the
top-8 for the two specific ⊆-shape variants. This is *probabilistic
forgetting* (top-k re-ordering), not deletion.

## Why this happened

1. NS11 evolved data contains **141 Nat rows / 11 Set rows / 0 Finset
   rows** (152 total). The Nat-heavy block re-anchors the distribution
   for `simp [...]`-style emissions toward Nat arguments.
2. Even though the combined corpus has 1,716 Set rows (∼30% of the
   total), the 141 new Nat examples are concentrated on very specific
   goal shapes (`⊢ … = …` Nat arithmetic, `⊢ … ↔ …` iff-omega
   chains). After 3 epochs at lr=1e-5, the model's `simp` argument
   distribution shifted globally — including for Set goals whose
   surface form happens to start with simp.

## v5 base corpus stats

- 5,577 total v5 rows
- 1,705 `Set.*` theorem rows (30.6%)
- 3,752 `Finset.*` theorem rows (67.3%)
- 120 `Nat.*` theorem rows (2.2%)
- 324 rows where `theorem=Set.subset_univ`; 288 where
  `theorem=Set.empty_subset`; 111 rows containing `Set.subset_def`
  in the tactic.

The v5 corpus already heavily oversamples `Set.subset_univ` /
`Set.empty_subset` — yet 3 epochs of fine-tune at lr=1e-5 still
managed to displace the winning tactic from top-8. The lift comes
from the 141 evolved Nat rows that didn't exist in v5; their gradient
signal is concentrated on Nat shapes.

## Implications for NS12

1. **Lower lr / fewer epochs** alone should help: 3 epochs at 1e-5 is
   ~6× the gradient updates of a 1-epoch run, and the per-example
   gradient on the 141 Nat additions is large relative to the v5
   distribution.
2. **Explicit replay** of `simp [Set.subset_def]` on `Set.subset_univ`
   and `Set.empty_subset` states would restore the top-k anchoring.
3. **Oversampling** Set rows in the combined corpus would dilute the
   Nat gradient.

NS12 will test all three.

## Files

- `project/evolve/eval_runs/gen_v5_raw_demo_v1/eval-1d29613c/metrics.json`
- `project/evolve/eval_runs/gen_v5_ns11_combined_raw_demo_v1/eval-0035102a/metrics.json`
- `project/evolve/eval_runs/gen_v5_ns11_combined_raw_demo_v1/eval-0035102a/traces.jsonl`
- `project/evolve/reports/ns11_learn_scale_report.md`
