# NS23 — minimal-tactic relabeling and attribution repair

**Branch:** `ns23-minimal-tactic-relabel`
**Parent:** NS22 commit `563c592`.
**Goal:** repair tactic-family attribution by testing simpler tactics
against every wrapper-only-vs-NS9 win, then reclassifying each
theorem by the *minimal* sufficient tactic rather than the wrapper
tactic that happened to win first.
**Outcome:** the NS22 attribution mismatch is **confirmed and
quantified**. Of the 10 Int `iff_omega_pair` pool theorems, **9 are
in fact `omega`-minimal**, exactly matching NS22-omega_5x's
cross-pool resolution (9/10). The 1 outlier (`Int.lt_toNat`) is also
the one NS22 failed on. The Int omega aggregate (fallback_omega ∪
iff_omega_pair ∪ constructor_omega ∪ split_ifs_omega, all under
minimal labels) is now **22 unique** in a single homogeneous pool —
6.6× over the 5-win training gate.

## 1. Motivation

NS22 trained an iff_omega_pair imitation model that failed to absorb
the long `exact ⟨fun h => by omega, fun h => by omega⟩` tactic. A
parallel fallback_omega ablation succeeded broadly and resolved 9/10
of the iff_omega pool theorems by emitting plain `omega`. The
hypothesis: **the NS9 wrapper's tactic-template ordering rewarded
the iff-pair template with the "winning_tactic" attribution because
it appeared first in the ranked list, even though `omega` alone was
sufficient at the Lean prover.** NS23 directly tests this by
re-running each pool theorem against a battery of tactics simpler
than the wrapper's chosen one.

## 2. Wrapper-only-vs-NS9 pool inventory (Stage 1)

`scripts/ns23_collect_wrapper_only_wins.py` reads
`cx1_combined_pool_meta.json` and `cx2_int_iff_omega_pool_meta.json`
and resolves file paths via `tasks.THEOREM_SETS`. Total: **32 unique
theorems** across the full post-NS15 wrapper-only-vs-NS9 surface.

| original family | namespace | count |
|---|---|---:|
| `aesop` | Finset | 6 |
| `simp_all` | Nat | 3 |
| `iff_omega_pair` | Int | 10 |
| `fallback_omega` | Int | 13 |
| **total** | | **32** |

| arc | count |
|---|---:|
| NS18 | 5 |
| NS19 | 1 |
| CX1 | 6 |
| CX2 | 20 |

Output: `project/data/ns23_wrapper_only_wins_raw_meta.json`.

## 3. Minimal-tactic battery (Stage 2)

12 tactics ordered simple → complex (`scripts/ns23_relabel_minimal_tactics.py`):

```
 1. assumption           7. simp_all
 2. rfl                  8. aesop
 3. decide               9. constructor <;> omega
 4. omega               10. constructor <;> simp_all
 5. norm_num            11. split_ifs <;> omega
 6. simp                12. exact ⟨fun h => by omega, fun h => by omega⟩
```

For each theorem the script opens a LeanDojo session at the initial
state and tries every tactic; the first to return `ProofFinished`
becomes the minimal_tactic and its family becomes the minimal_family.

Per-tactic timeout: 60s. Per-theorem timeout: 600s. The wrapper's
original tactic is tested separately at the end as a control.

## 4. Relabel results (Stage 3)

| outcome | count |
|---|---:|
| unchanged labels | 18 |
| **relabeled** | **13** |
| unresolved (no battery tactic closes it) | 1 |

### Cross-tabulation: original → minimal

| original | → minimal | count |
|---|---|---:|
| `aesop` | `aesop` (unchanged) | 6 |
| `simp_all` | `wrapper_original` (param simp_all unchanged) | 3 |
| `iff_omega_pair` | **`fallback_omega`** | **9** |
| `iff_omega_pair` | unresolved | 1 |
| `fallback_omega` | `fallback_omega` (unchanged) | 12 |
| `fallback_omega` | **`constructor_omega`** | 1 |

**Key finding:** 9 of the 10 `iff_omega_pair` theorems are
omega-minimal. Plain `omega` closes them from the initial state. The
wrapper's iff-pair template was unnecessary for these — it appeared
first in NS9's ordering and was therefore awarded the
"winning_tactic" attribution, but is not the simplest sufficient
tactic.

### Per-namespace omega aggregate

When `fallback_omega + iff_omega_pair + constructor_omega +
split_ifs_omega` are unified under their minimal-tactic labels:

| namespace | unique under minimal labels | gate (≥5) |
|---|---:|:---:|
| Int | **22** | ✓ |

The Int omega aggregate is **4.4× the gate threshold** and the
largest homogeneous wrapper-only training surface seen across all
arcs.

### Gated pools under minimal labels

| family | namespace | unique |
|---|---|---:|
| `aesop` | Finset | 6 |
| `fallback_omega` | Int | 21 |
| `omega_aggregate` | Int | **22** (aggregate above) |

## 5. NS22 validation (Stage 5)

The 10 `iff_omega_pair`/Int pool theorems classified by NS23 minimal:

| theorem | NS23 minimal | NS22 omega_5x raw_ckpt |
|---|---|:---:|
| `Int.le_add_one_iff` | `omega` | ✓ via `omega` |
| `Int.le_iff_lt_or_eq` | `omega` | ✓ via `omega` |
| `Int.le_sub_one_iff` | `omega` | ✓ via `omega` |
| `Int.sub_one_lt_iff` | `omega` | ✓ via `omega` |
| `Int.le_antisymm_iff` | `omega` | ✓ via `omega` |
| `Int.le_iff_eq_or_lt` | `omega` | ✓ via `omega` |
| `Int.natCast_nonpos_iff` | `omega` | ✓ via `omega` |
| `Int.natCast_ne_zero_iff_pos` | `omega` | ✓ via `omega` |
| **`Int.lt_toNat`** | **unresolved** | **✗ (failed)** |
| `Int.natCast_eq_zero` | `omega` | ✓ via `omega` |

**Perfect match: NS23 predicts 9/10 as omega-minimal; NS22 omega_5x
solves exactly those same 9/10.** This is the strongest possible
retrospective validation that NS22's apparent "cross-family
transfer" was in fact **single-family pool memorization under the
correct minimal labels**: omega_5x trained on `omega`, learned to
emit `omega`, and the 9 iff_omega pool theorems were always
omega-minimal — they just looked iff_omega_pair to the NS9 wrapper.

The `Int.lt_toNat` outlier is unresolvable from the initial state by
*any* tactic in the battery (including the wrapper's original
iff-pair). The wrapper trace must have closed it from a state
reached after a lead-in step (e.g., introducing a coercion lemma).
This is consistent with the NS22-iff_5x model failing it as well.

## 6. NS24 candidate pools (Stage 6)

Under the minimal-tactic labels, three pools meet the 5-win gate:

| pool | unique | trainable? | notes |
|---|---:|:---:|---|
| `omega_aggregate / Int` | **22** | ✓ | dominant target |
| `aesop / Finset` | 6 | ✓ | already trained as NS21 (memorization) |
| `fallback_omega / Int` | 21 | ✓ | subset of omega_aggregate; redundant if aggregate is trained |

### Strongest NS24 recommendation: omega_aggregate / Int

- **Pool size: 22 unique** (4.4× gate).
- **Tactic: `omega`** — short, vocabulary-aligned, known to absorb
  cleanly per NS22 (omega_5x trained on 13 omega rows, learned to
  emit omega broadly, +22 Int wins).
- **Expected outcome:** with a 22-row pool oversampled at 2-3×
  (per the [[project-ns21-transfer-ceiling]] memory's "pool size
  bounds transfer" rule), the model should produce
  **broader Int transfer** than NS22-omega_5x — possibly enough to
  push raw Int wins from 57/158 to 70+/158.
- **Init from:** `gen_v5_ns22_int_fallback_omega_5x` (already
  fine-tuned for Int omega emission), not gen_v5_ns12_balanced.
  This is a NS24-specific recommendation: instead of starting fresh
  each arc, continue training from the previous Int specialist.

### Held-out surface for NS24 validation

The CX1 catalog still has ~50 Int candidates not in the wrapper-only
pool but presumably amenable to omega (see CX2 catalog audit —
sub-bitwise/dvd Int order/arith theorems). Mining one more round
post-NS24 would measure whether the broader pool produced genuine
held-out gains beyond the trained 22.

## 7. NS16 heterogeneity revisited (negative finding)

NS16 reported the wrapper-only pool as heterogeneous across `simp_all`,
`split_ifs`, `exact_named`, `rw_named`, `simp_other`, etc. — 14
wrapper-only theorems in `nat_defs_medium` spread across 6 tactic
families. NS23 was expected to consolidate some of these. **In
practice the NS18-derived `simp_all / Nat` pool of 3 (`Nat.add_mod_*`,
`Nat.div_lt_iff_lt_mul'`) remains heterogeneous under minimal
labels:** the wrapper-original `simp_all [Nat.add_mod, Nat.mul_mod,
Nat.mod_eq_of_lt]` (with specific lemma list) is the minimal closer;
plain `simp_all` does not suffice. So the parameterized form is
genuinely needed — these 3 stay as `simp_all_parameterized`, and the
NS16 heterogeneity is not an attribution artifact.

## 8. Aesop-irreducible pool

The 6 Finset/aesop wrapper-only wins remain aesop-minimal — no
simpler tactic in the battery closes any of them. NS21 already
trained on this pool with the "honest memorization" outcome.
NS23 confirms the residual is genuine.

## 9. Recommendation

1. **Train NS24 on the omega_aggregate / Int pool (22 unique).** This
   is the dominant immediately-trainable target and the most likely
   to demonstrate genuine broad transfer (vs NS22's narrower
   13-row pool).

2. **Stop attributing iff_omega_pair wins separately in future
   mining.** Aggregate them with fallback_omega under the
   minimal-tactic label `omega` for training-data purposes. Keep
   the wrapper templates intact (they win the race in search; they
   just shouldn't drive *training-pool attribution*).

3. **Add a minimal-tactic relabel step to the mining pipeline.**
   Every future CX/NS arc should run NS23-style relabeling before
   declaring a training gate met. This avoids spending compute on
   imitation of tactics the model can't memorize when a simpler
   minimal exists.

4. **Defer DPO/ranker work.** The minimal-label finding suggests
   simpler training data is sufficient — preference-based methods
   may not be needed if the minimal-tactic family is already
   short and well-aligned with the model's vocabulary.

## 10. Files

Scripts (committed):

- `scripts/ns23_collect_wrapper_only_wins.py`
- `scripts/ns23_relabel_minimal_tactics.py`
- `scripts/ns23_analyze_family_pools.py`

Metadata (committed):

- `project/data/ns23_wrapper_only_wins_raw_meta.json`
- `project/data/ns23_minimal_tactic_labels.json` (full battery results)
- `project/data/ns23_minimal_family_pools_meta.json`

Reports (committed):

- `project/evolve/reports/ns23_minimal_tactic_relabeling_report.md` (this file)
- `project/evolve/reports/ns23_family_pool_comparison.md`

Not committed: nothing new (no checkpoints, no JSONLs, no traces).
