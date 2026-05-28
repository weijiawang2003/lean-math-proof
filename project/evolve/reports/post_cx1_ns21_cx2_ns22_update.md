# Post-NS20 update — CX1 / NS21 / CX2 / NS22

A short bridge report covering the four arcs that follow the
`learn_track_final_report_ns10_ns20.md` convergence claim. The
one-line takeaway: NS20 was not the end of the Learn track, and the
single most useful new principle is to **train on the shortest
sufficient tactic family, not merely the wrapper-attributed family.**

## Why NS20 was not final framework convergence

NS16–NS20 concluded that no homogeneous wrapper-only pool of ≥5
theorems existed outside the already-distilled iff-omega family. That
conclusion was correct *for the catalog at the time* — 527 theorems
drawn from three Mathlib source files (Nat/Defs, Set/Basic,
Finset/Basic), of which Nat and Finset were fully exhausted. The
ceiling was **old-catalog exhaustion, not framework exhaustion**. The
pipeline (mine wrapper-only wins → distill homogeneous pool → route)
was never the bottleneck; the supply of fresh theorems was.

## CX1 — catalog extension reopened the loop

CX1 scanned 44 additional Mathlib files and grew the catalog **527 →
1,817 available theorems across 8 namespaces**, with Int, Option, and
Bool entirely new. A limited eval probe found 6 truly-new wrapper-only
wins, and the **aesop/Finset pool reached 6 unique** (`coe_insert`,
`cons_eq_insert`, `disjUnion_singleton`, `coe_cons`,
`card_insert_eq_ite`, `image_id`, all winning tactic `aesop`),
meeting the NS21 training gate. CX1 proved the loop was supply-limited,
not converged.

## NS21 — Finset/aesop imitation, mostly memorization

NS21 trained `gen_v5_ns21_finset_aesop_20x` on the 6-theorem aesop pool
and routed `^Finset\.` to it. The verdict was **honest memorization /
narrow imitation**: 5/6 pool theorems solved raw with `aesop`, but
**0 held-out gains**, because NS12 already emitted `aesop` on the
held-out Finset surface. Local gains only (`ns17_finset_extra` 12→15,
`cx1_finset_image_filter` 28→30); all other routed-raw and wrapper
baselines preserved. Lesson reinforced: gains require a **fresh
namespace** the base model has no prior on — which pointed at Int.

## CX2 — Int pools found at a high strike rate

CX2 extended the Int catalog **120 → 216 candidates** (78 fresh after
exclusions) and ran the unmodified NS9 wrapper. **Wrapper-only strike
rate: 20/78 = 26%** — an order of magnitude above CX1's Bool/Option/Int
probe — confirming Int as genuinely underserved by the NS12 base model.
Two homogeneous gates were met without any experimental wrapper:

- `iff_omega_pair` / Int — **10 unique**, all
  `exact ⟨fun h => by omega, fun h => by omega⟩`.
- `fallback_omega` / Int — **13 unique**, all bare `omega`.

## NS22 — Int omega transfer, +22 raw Int wins

NS22 trained three variants from `gen_v5_ns12_balanced`. The
`iff_omega` candidates (5× and 10× oversample) failed to memorize the
long iff-pair tactic and produced ~0 net Int lift. The
`fallback_omega_5x` variant — built as an ablation — became the chosen
route. Routing `^Int\.` to it adds **+22 raw Int wins (NS12 baseline
35 → 57)** across the CX1+CX2 Int suite. It solved **13/13** of the
`fallback_omega` pool and **9/10** of the `iff_omega` pool **without
ever seeing an iff_omega theorem in training** — cross-family transfer
via the short `omega` tactic. Every Nat / Set / Finset / demo
routed-raw and wrapper baseline is preserved exactly; on Int the
wrapper now adds essentially zero incremental wins.

## New principle: shortest sufficient tactic > first wrapper-attributed tactic

NS22 exposed a **wrapper-attribution mismatch**. NS9's win attribution
awarded many Int goals to the `iff_omega_pair` template because it won
the race within the wrapper's ordering — but plain `omega` was in fact
sufficient for those same goals at test time (Lean's `omega` reflects
iff goals over linear-arithmetic predicates automatically). Two
consequences:

1. The 60M-param CodeT5-small base model cannot memorize a 49-character
   structured tactic from a small pool, even at 10× oversampling.
   **Short, vocabulary-aligned tactics transfer; long structured terms
   do not.**
2. Future mining/training should **aggregate `iff_omega_pair` and
   `fallback_omega` into a single minimal `omega` family whenever
   `omega` succeeds**, while keeping the wrapper templates intact for
   search-time win attribution.

## Recommended next: NS23 minimal-tactic relabeling

The next step is **not** another immediate training run but an
**attribution repair**: re-run wrapper wins through a minimal-
sufficient-tactic check and relabel each win by the shortest tactic
that actually closes it, then re-derive the training pools. Only after
labels are trustworthy should the next training arc (or stronger
wrapper capabilities) proceed. Lower-priority follow-ups: CX3
Bool/Option `decide`-family mining, and a DPO/ranker objective for the
long structured tactics that simple imitation cannot absorb.

## Pointers

- CX1: `cx1_catalog_extension_report.md`
- NS21: `ns21_finset_aesop_training_report.md`, `ns21_transfer_analysis.md`
- CX2: `cx2_int_iff_omega_mining_report.md`, `cx2_pool_summary.md`
- NS22: `ns22_int_iff_omega_training_report.md`, `ns22_transfer_analysis.md`
- Prior arc: `learn_track_executive_summary.md`,
  `learn_track_final_report_ns10_ns20.md`
- NS22 router: `project/evolve/routing/ns22_router.json`
