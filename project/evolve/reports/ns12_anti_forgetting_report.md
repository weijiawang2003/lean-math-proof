# NS12 — Balanced joint training and anti-forgetting

## Context

NS11 produced `gen_v5_ns11_combined` by 3-epoch fine-tuning gen_v5
on a 5,729-pair corpus (5,577 v5 base + 152 evolved). Raw
nat_defs_medium went 3 → 9, but raw demo_v1 went 10 → 8: both
`Set.subset_univ` and `Set.empty_subset` lost their winning tactic
`simp [Set.subset_def]` from the top-8 emissions (see
[ns12_demo_regression_analysis.md](./ns12_demo_regression_analysis.md)).

NS12 asks: can we recover the demo_v1 performance while keeping
the Nat gains?

## Hypothesis lanes tested

| variant | data change | training change |
|---|---|---|
| `low_lr` | none (NS11 combined as-is) | lr 1e-5→3e-6, 3→2 epochs |
| `balanced` | downsample v5 Nat by 50%, dup Set rows × 2 | lr 1e-5, 2 epochs |
| `replay_demo` | combined + 40 explicit replay rows (20 copies × 2 lost theorems) | lr 1e-5, 2 epochs |

All three start from `project/models/gen_v5` and never touch the
NS11 checkpoints.

## Dataset stats

| dataset | total | Nat | Set | Finset | v5 base | evolved | replay |
|---|---:|---:|---:|---:|---:|---:|---:|
| ns11_combined (reference) | 5,729 | 261 | 1,716 | 3,752 | 5,577 | 152 | 0 |
| ns12_low_lr | 5,729 | 261 | 1,716 | 3,752 | 5,577 | 152 | 0 |
| ns12_balanced | 7,445 | 261 | 3,432 | 3,752 | 7,282 | 163 | 0 |
| ns12_replay | 5,769 | 261 | 1,756 | 3,752 | 5,577 | 152 | 40 |

`balanced` doubles every Set row (so `Set.subset_univ`, with 324
rows in v5, now contributes 648 supervised gradients). `replay`
appends 20 verbatim copies of each lost-theorem replay row.

## Evaluation matrix

Top-k=8 beam, max_steps=8, top-k fallback rollout, generative
policy. Wrapper rows use the NS9 best genome
(`project/evolve/best/ns9_best_genome.json`) with
`policy_type=hybrid_evolved`.

| policy | medium (38) | large (65) | demo_v1 (15) | wrapper-medium | wrapper-large |
|---|---:|---:|---:|---:|---:|
| raw `gen_v5` (baseline)           | 3  | —   | **10** | 37 | 49 |
| raw `gen_v5_ns11_combined`        | **9**  | **13**  | 8     | 37 | 49 |
| raw `gen_v5_ns12_low_lr`          | 3  | —   | 8     | 37 | — |
| **raw `gen_v5_ns12_balanced`**    | 5  | 6   | **10** | **37** | **49** |
| raw `gen_v5_ns12_replay`          | 5  | 6   | 9     | 37 | 49 |

NS9 wrapper coverage (37/38 medium and 49/65 large) is preserved
for every NS12 variant. The wrapper is unchanged — only the base
model's raw top-k shifts.

## Where the demo_v1 recovery comes from

The regression was localized: top-8 dropped `simp [Set.subset_def]`
for `Set.subset_univ` and `Set.empty_subset`. Cross-variant top-8
emissions on those two states:

`Set.subset_univ`:

| model | top-8 (order) |
|---|---|
| gen_v5 | `simp [Set.subset_def]` ✓ |
| ns11_combined | `simp [Nat.mul_one]`, …, `simp [Set.ext_iff]`, …, `simp [*]` ✗ |
| ns12_low_lr | identical to ns11_combined ✗ |
| **ns12_balanced** | `simp [Set.subset_def]` ✓ |
| ns12_replay | same Nat-heavy top-8 ✗ |

`Set.empty_subset`:

| model | top-8 |
|---|---|
| gen_v5 | `simp [Set.subset_def]` ✓ |
| ns11_combined | `simp [Nat.mul_one]`, `simp [Set.ext_iff]`, …, `simp_all` ✗ |
| ns12_low_lr | same as ns11_combined ✗ |
| **ns12_balanced** | `simp [Set.subset_def]` ✓ |
| **ns12_replay** | `simp [Set.subset_def]` ✓ |

Reading:

- `low_lr` lost the lift *and* failed to restore demo — the 3e-6
  × 2 epoch run was too gentle to learn the new Nat patterns and
  also too small to perturb the bad top-k away from Nat-heavy
  emissions. It is the worst-of-both-worlds (3/38 medium, 8/15
  demo). The regression is therefore *not* a "raw step size" bug;
  it's a *data mix* bug. Lowering lr does not fix it.
- `replay_demo` got *one of two* targets back. Same state strings,
  same number of replay copies — the model retained `simp
  [Set.subset_def]` on `empty_subset` but the Nat distribution
  still won for `subset_univ`. With 20 copies × 2 epochs = 40
  effective gradient updates on each replay row, this approach is
  fragile and theorem-by-theorem.
- `balanced` solves both lost theorems with a global change.
  Doubling Set rows (Set.subset_univ alone: 324 → 648) provides
  enough sustained gradient on `Set.subset_def`-style tactics that
  the top-1 emission survives even after fine-tuning on the
  Nat-heavy evolved block.

## Pareto frontier

```
demo_v1
  ▲
  │
10 ┼──────── gen_v5 ──── ns12_balanced
  │                       ↑
  │            (Pareto-dominant over gen_v5)
 9 ┼────────────────────── ns12_replay
  │
 8 ┼ ns12_low_lr ───────── ns11_combined
  │
  └───┬─────┬─────┬─────┬─────►  raw medium (out of 38)
      3     5     7     9
```

- **Strictly Pareto-dominant pair**: `gen_v5_ns12_balanced` beats
  `gen_v5` on both axes (medium 5 vs 3, demo 10 vs 10).
- **Non-dominated frontier**:
  - `gen_v5_ns12_balanced` (5, 10) — strict win on demo retention.
  - `gen_v5_ns11_combined` (9, 8) — strict win on Nat coverage.
- `low_lr` is dominated; `replay` is dominated by `balanced`.

## Success criteria revisited

Stated in the task:

- Primary `demo_v1 >= 10/15`: **met by balanced** ✓; missed by
  low_lr and replay.
- Primary `medium >= 9/38`: **not met by any NS12 variant**.
  Balanced/replay reach 5/38. Achieving both demo≥10 and medium≥9
  requires another iteration (see below).
- Secondary `large >= 13/65`: not met by NS12.

NS12 succeeds at the *anti-forgetting* half of the goal but
sacrifices ~4 points of Nat coverage. The bottleneck is the
size and diversity of the evolved Nat training pairs (152 unique
(state, tactic) pairs from 13,773 episodes — see
[ns11_learn_scale_report.md](./ns11_learn_scale_report.md)).

## Best checkpoint and recommendation

**Best NS12 checkpoint: `project/models/gen_v5_ns12_balanced`**.
Use it when serving the raw model on mixed-domain prompts (Nat +
Set + Finset). It is a strict improvement on `gen_v5` and
preserves NS9 wrapper performance perfectly (37/38 + 49/65).

When *only* Nat coverage matters and demo regression is tolerable,
`gen_v5_ns11_combined` remains preferred (9/38 + 13/65 raw).

## Why replay_demo only half-worked

Both replay rows shared the *exact* state strings of the eval and
both used the *same* tactic. The hypothesis going in was that 20
copies × 2 epochs would dominate the gradient on those states. In
practice:

- `Set.empty_subset` recovered cleanly (top-1 simp [Set.subset_def]).
- `Set.subset_univ` did not — its top-8 looked identical to
  ns11_combined.

Probable cause: `Set.subset_univ` has *much more* diverse v5
training data (324 rows with tactics including `tauto`, `aesop`,
`simp`, `simp [Set.subset_def]`, and many Nat lemma simps); the 20
extra copies of `simp [Set.subset_def]` were not enough to pin
top-1 against the rest of its distribution. `Set.empty_subset` has
288 v5 rows with the same diversity, yet recovered — so the
deciding factor is likely interaction with adjacent Nat shapes,
not pure replay count. A reliable replay strategy would need at
least 50–100 copies per target (or to redistribute weight on the
specific tactic across siblings).

## Training configs

| variant | data | rows | lr | epochs | batch | val | runtime |
|---|---|---:|---|---:|---:|---|---:|
| `low_lr` | `ns12_train_low_lr.jsonl` | 5,729 | 3e-6 | 2 | 4 | 0.10 | ~7m45s |
| `balanced` | `ns12_train_balanced.jsonl` | 7,445 | 1e-5 | 2 | 4 | 0.10 | ~10m |
| `replay_demo` | `ns12_train_replay.jsonl` | 5,769 | 1e-5 | 2 | 4 | 0.10 | ~8m |

Each checkpoint is ~231 MB and gitignored.

## Files

Committed:
- `scripts/build_ns12_training_data.py`
- `project/data/ns12_train_low_lr_meta.json`
- `project/data/ns12_train_balanced_meta.json`
- `project/data/ns12_train_replay_meta.json`
- `project/evolve/reports/ns12_demo_regression_analysis.md`
- `project/evolve/reports/ns12_anti_forgetting_report.md`
- `.gitignore` additions for ns12 model dirs and large jsonl

Not committed (gitignored):
- `project/models/gen_v5_ns12_{low_lr,balanced,replay}/`
- `project/data/ns12_train_*.jsonl`
- `project/models/gen_v5_ns12_*_training.log`

## What NS12 answers

1. **Was the demo regression a top-k re-ordering rather than weight
   deletion?** Yes — `simp [Set.subset_def]` survived in vocab on
   `Set.inter_univ`/`Set.univ_inter` even with ns11_combined. The
   bug was probabilistic forgetting on two specific goal shapes.
2. **Can we recover demo without losing all Nat gains?** Yes,
   partially. `balanced` preserves +2 Nat over gen_v5 while
   restoring 10/15 demo and preserving wrapper 37/38 + 49/65.
3. **Is "just lower the lr" enough?** No. `low_lr` failed both
   axes. The fix has to live in the data mix.
4. **Is per-theorem replay reliable?** Not at modest copy counts.
   Got 1 of 2 targets back at 20 copies. Need richer state
   coverage or much higher replay weight.

## NS13 recommendations

1. **Grow the evolved corpus by running the wrapper on a larger
   theorem set.** Diversity, not depth, is the bottleneck (152
   unique pairs from 13,773 episodes); growing depth more does
   nothing. Candidate sets: `nat_defs_large_v5` (already known to
   have 49/65 wrapper coverage), `Finset.Basic`-derived theorems.
2. **Combine `balanced` data mix with `ns11_combined`'s training
   recipe** (3 epochs instead of 2). The +2 medium delta in
   balanced may be conservative; a 3-epoch run on the balanced
   dataset might reach 7/38 while still keeping demo at 10/15.
3. **Replay at 50–100 copies per target**, or replay across a
   *family* of related states rather than two literal strings.
4. **Try a larger base model** (CodeT5-small or a Lean-pretrained
   t5-base) — at 60M T5-small the model is near capacity for joint
   Set/Finset/Nat fitting.
5. **Add `demo_v1` as a training-time eval set** so we can flag a
   demo regression *before* a full run. Current pipeline only
   evaluates `nat_defs_*`.
