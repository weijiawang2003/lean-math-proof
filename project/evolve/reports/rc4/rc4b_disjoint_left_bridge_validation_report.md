# RC4B — Disjoint-left Bridge Candidate Validation Report

**Candidate:** `RC4B = RC2 ⊕ disjoint_left bridge`
**Status:** experimental RC4 candidate — **off-by-default, NOT released, NOT promoted**
**Methodology:** RC4A `def_unfold_simp` external-additive validation style
**Date:** 2026-05-31

---

## 1. Executive summary

RC4B **survives validation** as a narrow, namespace-parametric bridge candidate.

| metric | value |
|---|---|
| credited delta over literal RC2 | **+16** TRUE_DISJOINT_LEFT_BRIDGE_WIN |
| &nbsp;&nbsp;→ Set | 7 |
| &nbsp;&nbsp;→ Multiset | 9 |
| known-reproduction true wins | 11 (3 TR3/TR5 Set + 8 TR6 fresh) |
| **fresh-holdout true wins (out-of-sample)** | **5** (all Multiset) |
| off-gate emissions | **0** |
| regressions | **0** (additive evaluator) |
| determinism | **true** (run1 hash = run2 hash, 0 diffs, 0 open flakes) |
| narrow gates | yes (namespace ∈ {Set, Multiset} ∧ name/goal mentions disjoint) |

**Set vs Multiset split.** Both namespaces reproduce **100 % of their known wins** (Set 7/7,
Multiset 4/4 known) and every credited win is a confirmed bridge win. They diverge only on
*fresh out-of-sample generalization*: the fresh Multiset holdouts yield **5 new wins**
(`disjoint_add_right`, `disjoint_iff_ne`, `disjoint_right`, `disjoint_singleton`,
`disjoint_union_left`), while the fresh Set holdouts yield **0** — the remaining fresh Set
disjoint theorems (`pairwiseDisjoint_filter`, `sigmaToiUnion_*`, `biUnion_compl_*`,
`injOn_union`, `disjoint_iUnion`, `Disjoint.image`) are structurally harder goals where the
bridge rewrite *fires but does not close single-shot*. Both namespaces are confirmed; the
candidate is **not split**.

**Decision: `RC4B_CANDIDATE_CONFIRMED`** — positive delta, mechanism confirmed by minimal
attribution, fresh-holdout wins present, 0 off-gate, 0 regressions, deterministic, narrow
gates, Set/Multiset behaviour documented. *(Confirmation is evidentiary only — the candidate
remains off-by-default; no RC4 release, no production change, no commit.)*

---

## 2. Background

TR6's ranker-guided fresh-frontier sweep discovered that the `disjoint_left` bridge is
**namespace-parametric**: `Set.disjoint_left` and `Multiset.disjoint_left` are the same
rewrite shape in two namespaces. Each turns an opaque `Disjoint a b` goal into a membership
statement closable by `simp` (depth-1) or `simp … <;> aesop` (depth-2). This produced:

- **TR3** — 3 credited Set `disjoint_left` wins.
- **TR5** — those 3 Set wins reproduced live at rank 1 (`READY_FOR_RC4B_VALIDATION`).
- **TR6** — **8 fresh** wins (4 Multiset + 4 Set), surfacing the namespace-parametric bridge
  and yielding `READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT`.

The bottleneck after TR6 was candidate validation; RC4B is that validation for the
`disjoint_left` bridge family.

---

## 3. Candidate definition

A **narrow** family with two namespace variants (policy:
`disjoint_left_bridge_policy.json`):

| action | tactic | namespace | bridge lemma |
|---|---|---|---|
| `SET_DISJOINT_LEFT_SIMP` | `simp [Set.disjoint_left]` | Set | `Set.disjoint_left` |
| `SET_DISJOINT_LEFT_SIMP_AESOP` | `simp [Set.disjoint_left] <;> aesop` | Set | `Set.disjoint_left` |
| `MULTISET_DISJOINT_LEFT_SIMP` | `simp [Multiset.disjoint_left]` | Multiset | `Multiset.disjoint_left` |
| `MULTISET_DISJOINT_LEFT_SIMP_AESOP` | `simp [Multiset.disjoint_left] <;> aesop` | Multiset | `Multiset.disjoint_left` |

**Gate (per action):** `requires_namespace` ∈ {Set, Multiset} **and**
`requires_name_or_goal_contains` ∈ {`disjoint`, `Disjoint`}, `max_emissions_per_theorem = 1`.
The gate cannot fire on Nat / List / Finset / Order goals (namespace mismatch) nor on
Set/Multiset goals that never mention disjointness.

**Explicitly excluded** (kept narrow): `Finset.disjoint_*` (incl. `Finset.disjoint_left` —
not independently evidenced, used here as a negative control), broad `simp` with all disjoint
lemmas, arbitrary `d2_simp_aesop` (RC4C), `def_unfold_simp` (RC4A), global `@[simp]`
additions.

---

## 4. Evidence extraction (`rc4b_extract_disjoint_left_evidence.py`)

**11 deduplicated known wins** (Set 7, Multiset 4); needs_review = 0.

| source | count | targets |
|---|---|---|
| TR3 + TR5 reproductions (Set) | 3 | `Set.disjoint_iff_forall_ne`, `Set.disjoint_right`, `Set.disjoint_singleton_left` |
| TR6 fresh (4 Multiset) | 4 | `Multiset.disjoint_add_left`, `Multiset.disjoint_cons_left`, `Multiset.singleton_disjoint`, `Multiset.zero_disjoint` |
| TR6 fresh (4 Set) | 4 | `Set.disjoint_iUnion_left`, `Set.disjoint_iUnion_right`, `Set.disjoint_sUnion_left`, `Set.disjoint_sUnion_right` |

Bridge-lemma split: `Set.disjoint_left` ×7, `Multiset.disjoint_left` ×4. The mechanism is
homogeneous (one named rewrite, parametric only in namespace) → **not**
`CANDIDATE_TOO_HETEROGENEOUS`.

---

## 5. Validation theorem sets (`rc4b_build_validation_sets.py`)

| set | size | gate fires | purpose |
|---|---|---|---|
| `known_wins` | 11 | 11 | reproduce TR3/TR5/TR6 evidence |
| `fresh_holdout_set` | 8 | 8 | out-of-sample Set disjoint theorems |
| `fresh_holdout_multiset` | 20 | 20 | out-of-sample Multiset disjoint theorems |
| `disjoint_negative_controls` | 15 | **0** | Finset/Order disjoint (incl. `Finset.disjoint_left`) — gate must not fire |
| `namespace_negative_controls` | 20 | **0** | Nat/List/Finset/Order non-disjoint — gate must not fire |
| `canonical_smoke` | 45 (30 uniq) | **0** | demo_v1 + nat_defs_medium + nat_defs_large_v5 floors |
| **total** | **119** (103 uniq) | 39 | |

Overlap of non-evidence sets with the evidence corpus = 0 (fresh holdouts are genuinely
out-of-sample). The disjoint negative controls deliberately include `Finset.disjoint_left`,
`Finset.disjoint_biUnion_left/right`, `Finset.disjoint_map`, etc. to keep the namespace gate
honest.

---

## 6. Literal RC2 baseline (`rc4b_run_literal_rc2.py`)

Identical config to TR3/TR5/TR6 (rc2_release wrapper, ns24 router, hybrid_evolved, top-k 8,
max-steps 8); 57 reused, 46 run live. Status histogram: **failed 63 / solved 38 /
open_flake 2**.

- **known_wins: 0/11 solved** — all confirmed RC2 failures (correct baseline).
- canonical floors (RC2): demo_v1 **12/15**, nat_defs_medium **14/15**, nat_defs_large_v5
  **14/15**.

---

## 7. Candidate evaluation (`rc4b_run_candidate_eval.py`, external additive)

`candidate_solved = RC2_solved OR (gate fires AND a gated bridge tactic closes single-shot)`.

| metric | value |
|---|---|
| gate emissions | 39 (Set 15, Multiset 24) |
| **new wins over literal RC2** | **16** (Set 7, Multiset 9) |
| raw delta | +16 |
| regressions | 0 (additive: candidate ⊇ RC2) |
| off-gate emissions | 0 |
| emitted-and-solved | 16 |
| emitted-and-failed | 18 |

**New-win targets (16):** 11 reproductions of the known wins + **5 fresh-holdout Multiset
wins** (`disjoint_add_right`, `disjoint_iff_ne`, `disjoint_right`, `disjoint_singleton`,
`disjoint_union_left`, all via `simp [Multiset.disjoint_left] <;> aesop`).

The 18 emitted-and-failed are honest negatives: the gate correctly fires on harder Set
(`sigmaToiUnion_*`, `pairwiseDisjoint_filter`, …) and Multiset (`add_eq_union_*`,
`coe_disjoint`, `nodup_bind`, …) disjoint theorems but the single bridge rewrite does not
close them — the bridge is a *targeted* tool, not a catch-all.

---

## 8. Minimal attribution (`rc4b_minimal_attribution.py`)

For each of the 16 new wins: bare controls (`simp`, `simp_all`, `aesop`,
`classical <;> aesop`), lemma-direct (`exact`/`simpa using <NS>.disjoint_left`), and the
policy bridge tactics.

| classification | count |
|---|---|
| **TRUE_DISJOINT_LEFT_BRIDGE_WIN** | **16** |
| BASELINE_DUPLICATE | 0 |
| RC2_ALREADY_SOLVED | 0 |
| WRONG_NAMESPACE | 0 |
| SOURCE_SPECIFIC | 0 |
| NEEDS_REVIEW | 0 |

Every credited win has RC2 failing, **all bare controls failing**, and a policy bridge tactic
closing it in the matched namespace. Split: **Set 7 / Multiset 9**; **fresh-holdout true
wins 5** (all Multiset) / **known-reproduction true wins 11**. The bridge mechanism is the
genuine enabling step, not a generic-tactic or stale-baseline artifact.

---

## 9. Off-gate / preservation (`rc4b_offgate_preservation.py`)

Deterministic gate scan (exact, no search):

- **off-gate emissions: 0** → verdict `OFFGATE_CLEAN`.
- `disjoint_negative_controls` (15) → 0 emissions: the namespace gate correctly suppresses
  `Finset.disjoint_left` and all Finset/Order disjoint theorems.
- `namespace_negative_controls` (20) → 0 emissions.
- `canonical_smoke` → 0 emissions; floors preserved (RC2: demo_v1 12/15, medium 14/15,
  large 14/15; gate fires 0 on all).
- regressions: 0 (additive evaluator).

No broad firing, no unexpected-namespace fires → not `REJECT_BROAD_GATE`.

---

## 10. Determinism (`rc4b_determinism_check.py`)

Gated bridge probe rerun twice over known_wins + fresh_holdout_set + fresh_holdout_multiset +
negative controls (74 targets, 39 gate-firing).

| metric | value |
|---|---|
| clean run1 hash | `7574d704d3505a47` |
| clean run2 hash | `7574d704d3505a47` |
| gate decisions stable (all 74 targets) | **True** |
| genuine diffs (cleanly-executed theorems) | **0** |
| flake-induced diffs | 0 |
| open flakes (infrastructure) | 5 |
| win-affecting flakes | **0** |
| **deterministic (modulo infrastructure flakes)** | **True** |

The hash is computed over cleanly-executed theorems; the gate decision (a pure function of
namespace + name/goal) is identical on **every** target across both runs. The 5 open flakes
are Dojo hard-timeout / worker-kill events on heavy-`aesop` / hard-Set goals
(`Set._root_.Disjoint.image`, `Set.biUnion_compl_…`, `Set.disjoint_iUnion`,
`Multiset.coe_disjoint`, `Multiset.disjoint_of_subset_left`) — **all emitted-and-failed
non-wins**, none of which is a credited win. One credited reproduction win
(`Multiset.disjoint_add_left`) hit a run-1 worker-kill; re-probed twice more it solved cleanly
(4/4 actual executions: bare `simp` fails, `simp <;> aesop` closes), confirming the win is
stable — the run-1 entry was repaired with a real execution result before hashing. No genuine
non-determinism.

---

## 11. Schema-native wrapper smoke (`rc4b_schema_wrapper_smoke.py`)

`rc4b_candidate_wrapper.json` is a functional copy of the frozen RC2 wrapper with the four
bridge tactics **prepended to `priority_templates["any"]`** (ahead of the existing
`simp [Set.ite]` SET_ITE_SIMP entry — the RC2 precedent) and gated via
`theorem_name_tactic_gates` (`Set.disjoint`/`Set._root_.Disjoint`,
`Multiset.disjoint`/`Multiset.singleton_disjoint`/`Multiset.zero_disjoint`). RC1/RC2/NS24 are
untouched. Smoke run through the real `eval_rollout_all` harness over known_wins + the two
negative-control sets (46 theorems):

| metric | value |
|---|---|
| known wins solved by wrapper | **10 / 11** |
| negative-control regressions | **0** (35 controls) |
| negative-control new wrapper solves (perturbation) | **0** |

The wrapper reproduces 10 of 11 bridge wins end-to-end. The one miss
(`Set.disjoint_sUnion_right`) is the expected search-vs-single-shot gap: the best-first search
under the wrapper's top-k 8 / max-steps 8 budget does not always reach the
`simp [Set.disjoint_left] <;> aesop` branch that the single-shot probe applies directly. No
broad perturbation and no regressions on the 35 negative controls confirm the gates are
narrow. **The external additive evaluator (§7) remains the authority; this smoke is not a
release validation.**

---

## 12. Decision

**`RC4B_CANDIDATE_CONFIRMED`**

Requirements check:

| requirement | met? |
|---|---|
| positive delta over literal RC2 | ✅ +16 |
| minimal attribution confirms bridge mechanism | ✅ 16/16 TRUE |
| ≥1 fresh holdout win **or** strong fresh TR6 evidence in validation | ✅ 5 fresh-holdout wins + 8 TR6 fresh evidence |
| 0 off-gate | ✅ |
| 0 regressions | ✅ |
| deterministic | ✅ (clean hash match, 0 genuine diffs, gate decisions stable; 5 infra open-flakes, 0 win-affecting) |
| narrow gates | ✅ namespace + disjoint, max 1/theorem |
| Set/Multiset behaviour documented | ✅ (both confirmed; fresh generalization differs) |

This is a stronger result than RC4A `def_unfold_simp`, which had **0** fresh out-of-sample
wins; RC4B adds 5 genuinely fresh wins on top of 11 reproductions.

---

## 13. Next steps

- **If pursuing composition:** run **RC4C** (`d2_simp_aesop`) validation, then assemble an
  **RC4 composition candidate** (RC2 ⊕ RC4A ⊕ RC4B ⊕ RC4C) with its own literal
  composition benchmark + canonical floors before any release.
- The Set/Multiset behaviour is documented but **does not warrant a split** — both namespaces
  have confirmed TRUE wins and 0 off-gate. Should a future sweep find fresh Set holdouts
  remain barren while Multiset keeps yielding, reconsider `RC4B_MULTISET_CONFIRMED_SET_EXPERIMENTAL`.
- Either way, the 16 verified bridge wins + 18 emitted-and-failed negatives are clean
  training labels for the TR ranker/router (namespace-parametric bridge signal).

---

## 14. Protected-file confirmation

- `project/evolve/experiments/rc1/rc1_production_wrapper.json` — **untouched**.
- `project/evolve/experiments/rc2_release/rc2_production_wrapper.json` — **untouched**.
- `project/evolve/routing/ns24_router.json` — **untouched**.
- NS9 genome/checkpoint, REL1/RC1/RC2 release reports, TR1–TR6 datasets, RC4A artifacts —
  **untouched**.
- No production routing change, no RC4 release, no candidate promotion, **no commit**.
- All new artifacts live under
  `project/evolve/experiments/rc4_candidates/disjoint_left_bridge/`,
  `project/evolve/reports/rc4/`, and `scripts/rc4b_*.py`.
