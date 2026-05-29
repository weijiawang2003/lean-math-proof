# MX1 — live-Lean symbolic frontier mining

**Branch:** `mx1-live-symbolic-frontier-mining`
**Base:** SX1 (`63c2b78`)
**Stage:** signal acquisition (live LeanDojo; **no neural training**)
**Decision:** **negative for a new symbolic capability** — the production stack
already saturates the fresh Finset/Set frontier; the 2 new wins are
`aesop`-over-attributions, not symbolic labels. One cheap actionable follow-up.

---

## 1. SX1 recap

SX1 built a depth-2 symbolic *sequence* schema and planner; offline replay showed
the existing best-first search already performs the depth-2 follow-up, so a
fixed-battery sequence mode adds 0 net wins over production (Gate B). SX1's
recommendation was that the next real signal must come from **live mining**, not
offline replay — which is MX1.

## 2. Motivation

Acquire *new* clean symbolic labels and genuinely new wins by running the live
production-style stack (NS24 router + NS9 wrapper + WX3 Multiset oracle, AX4/SX1
off) over **fresh** theorem frontiers — not offline trace replay. Infrastructure
confirmed live: 18 GB traced Mathlib cache, `Dojo` opens in ~6 s, all routed
checkpoints present.

## 3. Frontier audit (Stage 1)

`project/data/mx1_symbolic_frontier_audit_meta.json`. Fresh = discovered catalog
(3989) minus every theorem consumed by prior CX/WX/AX/SX arcs (1799) and demo /
training sets, across the priority namespaces:

| namespace | fresh candidates | note |
|---|---|---|
| Set | 756 | ext-shaped (largest) |
| Finset | 606 | ext/cases |
| List | 237 | cases/induction |
| Multiset | 68 | mostly cross-surface `toFinset` |
| Option | 0 | **exhausted** by AX1/AX2/WX |

**Key:** the previously-mined Multiset/List/Option surfaces are essentially
exhausted (availability-screened fresh ≈ 1/0/0); the live symbolic frontier is
now **Finset and Set**, which require the *new* MX1 ext/cases actions.

## 4. Theorem sets (Stage 2)

`project/evolve/routing/mx1_theorem_sets.json` — 138 carved (bounded for live
mining; full frontier far larger, logged). `Multiset.eq_of_mem_map_const`
excluded (known REPL-hanger, from AX3). Registered via `tasks._load_mx1_sets()`.

| set | n |
|---|---|
| mx1_multiset_frontier | 18 |
| mx1_finset_symbolic_frontier | 40 |
| mx1_list_frontier | 20 |
| mx1_set_ext_frontier | 40 |
| mx1_mixed_symbolic_frontier | 20 |

## 5. New symbolic actions (Stage 4)

Added additively to `symbolic_actions.py` (+ Finset/Set coarse types in
`state_vars.py`): **`SET_EXT_SIMP`**, **`FINSET_EXT_SIMP`** (`ext x <;> simp[_all]`,
gated on a Set/Finset value), **`FINSET_CASES_SIMP`** (`cases {v} <;> simp[_all]`).
They flow through the existing AX1 wrapper path (`load_actions` +
`instantiate_symbolic_action`) — **no `strategy_wrapper.py` change**; disabled ⇒
byte-identical to NS9; off-gate ⇒ 0 emissions. Configs:
`mx1_set_finset_ext_safe.json`, `mx1_combined_symbolic_frontier_safe.json`.

## 6. Live mining variants (Stage 3)

`project/data/mx1_live_mining_probe_meta.json`. Variants A (routed raw), B (WX3
production wrapper), E (MX1 extended Set/Finset/Multiset symbolic), D (SX1 depth-2
sequence trace generator). C (AX4 predictor) is offline. ~2 s/theorem after
warmup.

| set | n | A raw | B prod | E sym | new>prod | sym fire (E) | sym-origin wins | regr |
|---|---|---|---|---|---|---|---|---|
| multiset_frontier | 18 | 3 | 4 | 4 | 0 | 32 | 1 | 0 |
| finset_symbolic_frontier | 40 | 10 | 10 | 10 | 0 | 140 | **0** | 0 |
| list_frontier | 20 | 5 | 5 | 5 | 0 | 0 | 0 | 0 |
| set_ext_frontier | 40 | 5 | 5 | **7** | **2** | 71 | 2 | 0 |
| mixed_symbolic_frontier | 20 | 10 | 10 | 10 | 0 | 44 | 0 | 0 |
| **total** | 138 | 33 | 34 | 36 | **2** | 287 | 3 | **0** |

Readings:
- **Multiset**: production already optimal — the induction action lands 1
  symbolic-origin win (B=4 vs A=3); E adds nothing beyond B.
- **Finset**: 140 ext/cases firings, **0 closes**. The new Finset actions never
  finish a goal; the routed NS21 finset-aesop policy already gets the 10
  winnable ones. The actions are inert here.
- **List**: the E config carries no List actions (Multiset+Set/Finset only), so
  E == B == NS9 (5 generative wins).
- **Set**: `SET_EXT_SIMP` (`ext x <;> simp`) closes **2 theorems production
  misses** — `Set.Finite.toFinset_insert`, `Set.Finite.toFinset_offDiag`.
- **0 regressions** everywhere.

## 7. Minimal relabel (Stage 5) — the decisive check

`project/data/mx1_minimal_symbolic_frontier_labels.json` (LIVE, strict battery
from the initial state). Both Set new wins classify as **`over_attributed_raw`**:
a plain **`aesop`** closes them directly — the `ext x <;> simp` symbolic win was
over-attribution. Production missed them only because `aesop` is not in the Set
route's emission/battery, not because a symbolic action was required.

⇒ **clean new single-shot symbolic labels = 0**; sequence-needed = 0.

## 8. Label pools (Stage 6)

`project/data/mx1_updated_symbolic_label_pools_meta.json`. Merged single-shot
pool total 73 (biggest family `MULTISET_INDUCTION_SIMP[Multiset,simp_all]` = 41);
**MX1 new clean labels = 0**. No new family reaches the training gate; the
sequence gate (≥20) remains unmet (biggest sequence family 3).

## 9. Preservation / runtime (Stage 7)

`project/data/mx1_preservation_matrix.json`. Static planner over preservation
initial states: **0 off-gate emissions** (Nat/Int/demo). Set/Finset/Multiset are
now gated families, so they emit there (additive, expected). Live regression
check (E vs B on the gated ns17 set/finset sets): **0 regressions**. Canonical
NS9 floors preserved by construction (genome byte-unchanged, additive ranked
list): medium 37/38, large 49/65, demo 11/15. Runtime ~2 s/theorem; per-theorem
timeout 900 s; the known REPL-hanger excluded.

## 10. Decision

**No new symbolic capability; do not train, do not promote the new actions.**

- **Train a new learner?** No — 0 new clean labels.
- **Promote a new wrapper action?** No — `FINSET_EXT_SIMP`/`FINSET_CASES_SIMP`
  never close (0/140); `SET_EXT_SIMP`'s 2 wins are `aesop`-over-attributed.
- **Mine further (same actions)?** Low yield — the routed generative policies
  already saturate the winnable fresh Finset/Set theorems; symbolic
  constructor/ext actions only paid off in the **Multiset quotient** namespace
  (WX3), where the base policy was weak. On Finset/Set the base policy is strong
  (NS21 aesop), so there is little symbolic headroom.
- **Revisit sequence learning?** No — gate unmet.

**One cheap actionable follow-up (a fallback tweak, NOT symbolic training):** the
2 genuinely-missed Set theorems are `aesop`-closable. Adding `aesop` to the Set
route's fallback/battery (additive, namespace-gated) would likely capture them
and similar `Set.Finite.toFinset_*` lemmas without any new symbolic label or
model — the same lesson as NS21 (aesop in the battery), now for Set. This is the
recommended next experiment if the Set surface is pursued.

**Net:** MX1 confirms the symbolic-action layer is **namespace-saturated** — its
value is concentrated where the base policy is weak (Multiset quotient), and the
fresh Finset/Set frontier is already covered by the routed generative policies.
The durable outputs are the frontier audit, the new (validated, off-by-default)
Finset/Set action types, and the precise negative result.
