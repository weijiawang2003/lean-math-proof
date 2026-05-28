# NS24 — Int minimal-sufficient omega aggregate training

**Branch:** `ns24-int-minimal-omega-training`
**Parent:** NS23 commit `021254a` (minimal-tactic relabeling).
**Goal:** test the NS23 hypothesis that training on the **shortest
sufficient tactic** (minimal label) beats training on the
wrapper-attributed tactic, by distilling the 22-theorem Int
`omega_aggregate` pool (21 `omega`-minimal + 1 `constructor <;> omega`)
into a stronger Int branch model and routing Int goals to it.
**Outcome (near-null / confirmatory):** the repaired minimal-omega
labels **reproduce** NS22's Int performance (57 → **58/156**, +1) but do
**not** unlock the hoped-for 65–70+ broad transfer. The relabeled
`iff_omega_pair` group is solved **9/9 by both NS22 and NS24** — direct
evidence that NS22's `fallback_omega` ablation had *already* absorbed the
`omega` policy, exactly as NS23 retrospectively predicted. The
minimal-tactic principle is validated as a **labeling-correctness** rule;
it did not add capability here because the omega surface was already
saturated by NS22.

## 1. NS23 recap

NS23 re-ran every wrapper-only-vs-NS9 win through a minimal-tactic
battery and relabeled each theorem by the simplest sufficient tactic:

- 32 wrapper wins relabeled; 13 changed family, 1 unresolved.
- **9 of 10** Int `iff_omega_pair` theorems are in fact `omega`-minimal
  (plain `omega` closes them from the initial state); the wrapper's long
  `exact ⟨fun h => by omega, fun h => by omega⟩` template merely won the
  NS9 ordering race. The 10th (`Int.lt_toNat`) is unresolved.
- Unified under minimal labels, the **Int omega aggregate = 22 unique**
  (21 `omega` + 1 `constructor <;> omega`, the latter `Int.zero_le_ofNat`)
  — 4.4× the 5-win gate and the largest homogeneous training surface
  across all arcs.

## 2. Training-data construction (Stage 1)

`scripts/build_ns24_training_data.py` reads the NS23
`omega_aggregate_by_namespace` pool and, for each theorem, extracts the
**initial (minimum-`step`) proof state** from the NS9-wrapper Int traces
(`cx1_ns9wrap_*`, `cx2_ns9wrap_*`), pairing it with the **NS23 minimal
tactic** — never the wrapper-attributed tactic.

Why the *initial* state, not the close-row state: `Int.zero_le_ofNat`'s
wrapper proof was two-step — a `simp` lead-in normalised `0 ≤ ofNat n`
→ `0 ≤ ↑n` before `omega` closed it. Plain `omega` does **not** close
its initial state, so its minimal label is `constructor <;> omega`. The
model sees the initial state at inference time, so that is what we train.

| variant | pool | target tactic(s) | oversample | total rows | init |
|---|---|---|---:|---:|---|
| `ns24_int_minimal_omega_5x`             | 21 omega-minimal | `omega` | 5×  | 7,550 | gen_v5_ns22_int_fallback_omega_5x |
| `ns24_int_minimal_omega_10x`            | 21 omega-minimal | `omega` | 10× | 7,655 | gen_v5_ns22_int_fallback_omega_5x |
| `ns24_int_minimal_omega_plus_constructor_5x` | 21 omega + 1 constructor | `omega`, `constructor <;> omega` | 5× | 7,555 | gen_v5_ns22_int_fallback_omega_5x |
| `ns24_int_minimal_omega_5x_from_ns12` (ablation) | same as 5x | `omega` | 5× | 7,550 | gen_v5_ns12_balanced |

Pool provenance (per-row, ×oversample): of the 21 omega-minimal
theorems, **12** were already in NS22's omega training set
(original `fallback_omega`) and **9** are newly added by relabeling
(original `iff_omega_pair`). Replay: full `ns12_train_balanced.jsonl`
(7,445 rows). Metas committed at `project/data/ns24_*_meta.json`;
JSONLs gitignored.

## 3. Training (Stage 3)

All checkpoints: 3 epochs, batch 8, lr 5e-5, max_src 512, max_tgt 128,
seed 42 (`scripts/ns24_train.sh`). Three variants continue from the
NS22 Int specialist; the ablation starts from `gen_v5_ns12_balanced`.
Final losses: 5x eval_loss 0.388, 10x 0.334, +constructor 0.342,
from_ns12 0.414. ~15 min/model on MPS. No existing checkpoint was
overwritten.

## 4. Raw checkpoint evaluation (Stage 4)

Int surface (`--top-k 8 --max-steps 8`, raw single checkpoint):

| set | avail | NS22 omega_5x | NS24 omega_5x | NS24 omega_10x | NS24 +constructor | NS24 from_ns12 |
|---|---:|---:|---:|---:|---:|---:|
| `cx2_int_iff_omega_easy`   | 12 | 5  | 5  | 5  | 5  | 5  |
| `cx2_int_iff_omega_medium` | 3  | 2  | 2  | 2  | 2  | 2  |
| `cx2_int_order_arith`      | 49 | 15 | 15 | 15 | 15 | 15 |
| `cx2_int_mixed`            | 12 | 6  | **7** | **7** | **7** | **7** |
| `cx1_bool_option_int`      | 80 | 29 | 28 | 29 | 29 | 29 |
| **Int total**              | 156 | **57** | 57 | **58** | **58** | **58** |

The entire Int delta is `cx2_int_mixed` 6 → 7 (the held-out theorem
`Int.cast_ite`). `omega_5x` alone also lost `Bool.lt_iff` in
`cx1_bool_option_int` (29 → 28), netting 0. The three other variants
gain `Int.cast_ite` with no loss → 58.

Preservation (raw `omega_5x` checkpoint, for reference — note Set/Finset
goals route elsewhere in production):

| set | NS22 omega_5x | NS24 omega_5x | NS24 omega_10x |
|---|---:|---:|---:|
| `demo_v1`              | 10/15 | 11/15 | 11/15 |
| `ns17_set_extra`       | 19/30 | 18/30 | 17/30 |
| `ns17_finset_extra`    | 12/30 | 14/30 | 14/30 |
| `ns14_set_finset_extra`| 11/20 | 13/20 | 12/20 |

These raw cross-domain numbers fluctuate ±1–2 between Int checkpoints,
but they are **not** the production preservation metric: in the router,
Set goals go to `gen_v5_ns12_balanced` and Finset to
`gen_v5_ns21_finset_aesop_20x`, never to the Int checkpoint. Routed
preservation (§7) is what matters and is preserved by construction.

## 5. Best checkpoint (Stage 5)

Three variants tie at 58/156 Int (+1 over NS22). Per the score
(Int net wins − penalties), all three strictly dominate NS22 on Int
(no Int losses). The chosen Int checkpoint is
**`gen_v5_ns24_int_minimal_omega_10x`**:

- It is the **homogeneous omega family** (respecting the hard "no broad
  mixed-family training in the main experiment" constraint), unlike
  `+constructor` which mixes a second tactic form.
- It ties for best Int (58) and shows no held-out Int losses vs NS22.
- The `from_ns12` ablation also reaches 58, showing the +1 does not
  require continuing from the NS22 specialist — i.e. the larger repaired
  21-omega pool (vs NS22's 13) is what nudges +1, not the warm start.

This is a **marginal** promotion. The headline target (65–70+) was not
met; NS24 confirms NS22 had already converged on the omega surface.

## 6. Router (Stage 6)

`project/evolve/routing/ns24_router.json` changes **only the Int route**
vs the NS22 router: `^Int\.` → `gen_v5_ns24_int_minimal_omega_10x`.
Nat → `gen_v5_ns15_nat_oversample`, Finset → `gen_v5_ns21_finset_aesop_20x`,
Set/default → `gen_v5_ns12_balanced` are unchanged, so Nat/Set/Finset
preservation carries over by construction.

## 7. Routed evaluation (Stage 7)

Routed raw (`ns24_router`, `--top-k 8 --max-steps 8`):

| set | NS24 routed | floor | met? |
|---|---:|---:|:---:|
| `nat_defs_medium`        | 23/38 | 23 | ✓ |
| `nat_defs_large_v5`      | 35/64 | 35 | ✓ |
| `demo_v1`                | 10/15 | ≥10 | ✓ |
| `ns17_set_extra`         | 18/30 | ≥18 | ✓ |
| `ns17_finset_extra`      | 15/30 | ≥15 | ✓ |
| `ns14_set_finset_extra`  | 13/20 | ≥13 | ✓ |
| **Int total (5 sets)**   | **58/156** | >57 | ✓ (+1 vs NS22) |

(`nat_defs_large_v5` reported 64 available this run — one theorem
transiently unavailable in LeanDojo; 35 proved meets the 35/65 floor.)
Every Nat/Set/Finset/demo floor is met exactly — as expected, since the
NS24 router changes only the Int route. Routed Int is 58, matching the
raw `omega_10x` total: Int goals route to `omega_10x`.

## 8. Wrapper compatibility (Stage 8)

NS9 best genome + `ns24_router` (`hybrid_evolved`):

| set | wrap + NS24 router | NS9 wrap baseline | met? |
|---|---:|---:|:---:|
| `nat_defs_medium`        | 37/38 | 37/38 | ✓ |
| `nat_defs_large_v5`      | 49/64 | 49/65 | ✓ |
| `demo_v1`                | 11/15 | 11/15 | ✓ |
| `cx2_int_iff_omega_easy` | 5/12  | (routed raw 5; wrapper adds 0) | ✓ |
| `cx2_int_order_arith`    | 16/49 | (routed raw 15; wrapper adds 1) | ✓ |

The wrapper baselines are preserved exactly. On Int, the NS24 raw model
has absorbed `omega` so completely that the wrapper adds essentially
zero incremental wins (+0 easy, +1 order_arith) — identical to the NS22
wrapper picture, and consistent with the saturation finding.

## 9. Transfer vs absorption analysis (Stage 9)

`scripts/ns24_compare_minimal_omega_transfer.py` →
`project/evolve/reports/ns24_transfer_analysis.md`. Pool groups under the
repaired labels (NS22 → NS24 candidate, all variants identical here):

| group | size | NS12 | NS22 | NS24 |
|---|---:|---:|---:|---:|
| `old_ns22_omega` (orig. fallback_omega) | 12 | — | 12 | 12 |
| `relabeled_iff` (orig. iff_omega_pair)  | 9  | — | **9** | **9** |
| `constructor` (`Int.zero_le_ofNat`)     | 1  | — | 1 | 1 |

**The decisive row is `relabeled_iff`: NS22 already solved 9/9, NS24 also
9/9.** Adding those 9 theorems to NS24's training set as explicit `omega`
rows changed nothing, because NS22 had already learned to emit `omega`
on iff-form Int goals (the cross-family transfer NS22 reported, reframed
by NS23 as single-family omega absorption). `Int.zero_le_ofNat` is solved
in multi-step rollout by both (a normalising step then `omega`), so the
constructor training row was also redundant.

Held-out Int (not in the 22-pool): the only gain vs NS22 is
`Int.cast_ite` (cx2_int_mixed); `omega_5x` additionally lost `Bool.lt_iff`.
Emitted-tactic distribution on solved Int goals is dominated by `omega`
with a long `simp_all`/`aesop` tail (inherited from the NS12 replay).

**Classification (script verdicts):** `omega_5x` =
`reproduction_near_null` (+0 Int); `omega_10x` / `+constructor` /
`from_ns12` = `marginal_gain` (+1 Int, demo losses 0, held-out Int losses
0). The raw Set/Finset deltas (e.g. omega_10x `ns17_set_extra` −2) are
**routed away** — Set goes to `gen_v5_ns12_balanced`, Finset to
`gen_v5_ns21_finset_aesop_20x` — so they are excluded from the regression
signal. This is neither memorization-only (the pool was already solved
pre-NS24) nor held-out transfer (a single held-out gain) nor regression
(demo and routed preservation hold). The repaired labels were *already
satisfied* by NS22.

## 10. Comparison across arcs

| arc | pool | base-model prior | outcome |
|---|---|---|---|
| **NS15** Nat iff_omega | 5 unique | NS12 had no Nat iff_omega | **broad transfer** (medium 3→23, large 9→35) |
| **NS21** Finset aesop | 6 unique | NS12 already emitted aesop on Finset | **memorization only** (+5 raw, 0 held-out) |
| **NS22** Int omega (wrapper labels) | 13 omega | NS12 had ~0 Int competence | **broad absorption** (+22 Int, 9/10 iff via omega) |
| **NS23** minimal relabel | — | — | 9/10 iff are omega-minimal; aggregate 22 |
| **NS24** Int minimal-omega aggregate | 21–22 omega | **NS22 already omega-saturated** | **near-null (+1)**; repaired labels reproduce NS22 |

The NS24 lesson refines the NS23 minimal-tactic-attribution
principle: **minimal-tactic relabeling is the correct way to *attribute*
and *avoid wasted* training (it would have saved the NS22 iff_5x/iff_10x
runs), but it only yields new wins when the minimal family is one the
base model has not already absorbed.** Here, NS22's ablation had already
done the absorption, so the repaired labels had nothing left to teach.

## 11. Recommendation

1. **Promote the NS24 router (marginal).** It strictly dominates the
   NS22 router on Int (58 vs 57) at no preservation cost. The win is one
   theorem; treat NS24 primarily as confirmation of the minimal-tactic
   principle, not a capability jump.
2. **The Int omega surface is saturated.** Further omega-pool mining on
   the *current* catalog will not help — NS22+NS24 have converged. To get
   genuine Int gains, mine **fresh held-out Int** (the CX2 audit left
   ~50 sub-bitwise/dvd Int order/arith candidates unprobed) and measure
   transfer to theorems neither NS22 nor NS24 has seen.
3. **CX3 Bool/Option decide-family mining** is the highest-yield next
   direction: Bool (35) and Option (47) are fresh namespaces with no base
   prior, the setting where NS15/NS22-style absorption actually produced
   broad transfer.
4. **Keep the minimal-tactic relabel in the pipeline** as a pre-training
   gate (it is cheap and prevents wasted long-tactic imitation runs), but
   do not expect relabeling alone to add wins on an already-absorbed
   family.
5. **DPO/ranker remains deferred** — the minimal-label finding holds:
   the transferable tactics are short and already learnable by imitation.

## 12. Files

Scripts (committed):
- `scripts/build_ns24_training_data.py`
- `scripts/ns24_train.sh`
- `scripts/ns24_run_eval.sh`
- `scripts/ns24_compare_minimal_omega_transfer.py`

Configs (committed):
- `project/evolve/routing/ns24_router.json`

Metadata (committed):
- `project/data/ns24_int_minimal_omega_5x_meta.json`
- `project/data/ns24_int_minimal_omega_10x_meta.json`
- `project/data/ns24_int_minimal_omega_plus_constructor_5x_meta.json`
- `project/data/ns24_int_minimal_omega_5x_from_ns12_meta.json`
- `project/data/ns24_transfer_analysis.json`

Reports (committed):
- `project/evolve/reports/ns24_int_minimal_omega_training_report.md` (this file)
- `project/evolve/reports/ns24_transfer_analysis.md`

`.gitignore` extended with NS24 paths. Not committed: checkpoints,
training JSONLs, eval traces/logs, eval-run directories.
