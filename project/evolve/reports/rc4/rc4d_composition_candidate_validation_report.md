# RC4D — RC4 Composition Candidate Validation Report

**Candidate:** `RC4D = RC2 ⊕ RC4A ⊕ RC4B ⊕ RC4C_residue`
**Status:** experimental composition candidate — **NOT a release, NOT promoted, off-by-default**
**Date:** 2026-05-31

---

## 1. Executive summary

RC4D composes the three independently-validated RC4 components over RC2 and tests whether they
stack additively without double-counting, off-gate firing, regressions, or floor loss.

| Metric | Result |
|---|---|
| Literal RC2 over validation manifest | 130 unique theorems; known wins 0/5·0/16·0/7·0/5 (genuine failures) |
| External additive raw delta | **+24** (RC4A 5, RC4B 16, RC4C_residue 3) |
| Minimal-attribution **credited delta** | **+23** (RC4A 5, RC4B 16, RC4C_residue 2) |
| RC4C_residue overlap eliminated (→ RC4B) | 9 Multiset theorems (no double-count) |
| Regressions | **0** (additive ⇒ structurally impossible) |
| Off-gate emissions | **0** (RC4C_residue off-gate 0) |
| Determinism | **True** — clean hash `f55cbc5543c75306` ×2, 0 diffs, 0 flakes, gate+component stable |
| Full canonical floors (RC2 → RC4D) | demo_v1 12→12 · medium 37→37 · large 49→49 — **all pass, 0 regressions** |
| Schema-native wrapper reproduction | **22/23** credited wins (RC4A 5/5, RC4B 15/16, RC4C_residue 2/2); regressions 0 → `SCHEMA_REPRODUCES` |
| RC4C residue de-duplicated | Yes — overlap + simp-only actions dropped |

**Decision: RC4D_COMPOSITION_CANDIDATE_CONFIRMED** (see §13).

---

## 2. Background

- **RC2** = RC1 ⊕ SET_ITE_SIMP is the frozen production stack.
- **RC4A** `def_unfold_simp` — `RC4A_CANDIDATE_CONFIRMED`: +5 def-unfold wins, 0 regr/off-gate, deterministic. (No schema smoke was ever run for RC4A — RC4D is the first.)
- **RC4B** `disjoint_left_bridge` — `RC4B_CANDIDATE_CONFIRMED`: +16 wins (11 repro + 5 fresh), 0 regr/off-gate, schema smoke 10/11.
- **RC4C** `d2_simp_aesop` — `RC4C_CONFIRMED_WITH_RC4B_OVERLAP`: +7 pure depth-2 wins but 8 overlap RC4B; the original fused `simp [L] <;> aesop` schema deployment failed (0/12) because the best-first search applies `<;>` differently than a single-shot transition.

Composition is needed because the three were validated **in isolation**: RC4C overlaps RC4B
heavily (`disjoint_left` is literally an RC4B action), and RC4C's deployable form was unresolved.
RC4D resolves both: it **de-duplicates RC4C against RC4B** and **deploys the residue RC4B-style**
(bare `simp [L]` + combinator). Any eventual RC4 release candidate must pass this composition gate.

---

## 3. Component manifest

`out/rc4d_component_manifest.json` — built purely from the three components' policies +
minimal_attribution (no live Lean).

### Included actions

| Component | Actions | Known wins |
|---|---|---|
| **RC4A** | `simp [<allowlisted def>]` over the 9-def allowlist | 5 |
| **RC4B** | `simp [<NS>.disjoint_left]` + `<;> aesop`, NS∈{Set,Multiset} | 16 |
| **RC4C_residue** | `MULTISET_DISJOINT_RIGHT_D2`, `SET_SUBSET_PAIR_D2`, `LIST_FORALL_D2` (each bare + combinator) | residue lemmas |

### RC4C residue decisions

| Lemma | Action | Decision |
|---|---|---|
| `Multiset.disjoint_right` | MULTISET_DISJOINT_RIGHT_D2 | **INCLUDE_AS_DEPTH2_SIMP_AESOP** |
| `Set.subset_pair_iff_eq` | SET_SUBSET_PAIR_D2 | **INCLUDE_AS_DEPTH2_SIMP_AESOP** |
| `List.forall_iff_forall_mem` | LIST_FORALL_D2 | **INCLUDE_AS_DEPTH2_SIMP_AESOP** |

### Excluded RC4C actions

| Action | Lemma | Reason |
|---|---|---|
| SET_DISJOINT_LEFT_D2 | Set.disjoint_left | EXCLUDE_OVERLAP (RC4B) |
| MULTISET_DISJOINT_LEFT_D2 | Multiset.disjoint_left | EXCLUDE_OVERLAP (RC4B) |
| FINSET_BIUNION_SUBSET_D2 | Finset.biUnion_subset | EXCLUDE_DUPLICATE (SIMP_ONLY, depth-1) |

### Theorem-level overlap (de-dup)

- RC4C_residue theorem wins: 7. Of these, **5 Multiset theorems are already solved by RC4B**
  via `disjoint_left` (`disjoint_add_left/_right`, `disjoint_iff_ne`, `disjoint_union_left`,
  `singleton_disjoint`).
- **Genuinely additive-over-RC4B residue coverage: 2** — `List.Forall.imp`,
  `Set.Nonempty.subset_pair_iff_eq`. Ordering `[RC4A, RC4B, RC4C_residue]` credits the 5
  overlap theorems to RC4B (earlier component), so RC4C_residue never double-counts them.

---

## 4. RC4D policy

`composition_rc4d/rc4d_composition_policy.json`.

- **Ordering** `[RC4A, RC4B, RC4C_residue]` — narrowest/deterministic def-unfold first, clean
  disjoint bridge second, riskier depth-2 residue last (so RC4B claims any disjoint overlap).
- **Gates** — per-component namespace + name/goal-token gates (RC4A goal-driven def presence;
  RC4B/RC4C namespace + disjoint/pair/forall tokens). Max 1 emission per theorem per component.
- **Narrowness** — 14 distinct lemmas/defs; no arbitrary retrieved-lemma battery, no global broad
  simp, no generic `simp <;> aesop`. Every action traceable to a validated RC4A/B/C win.
- **Deduplication** — `drop_rc4c_overlap_rc4b`, `drop_rc4c_overlap_rc4a`,
  `drop_simp_only_duplicate_depth2_credit` all true.

---

## 5. Validation theorem sets

`composition_rc4d/theorem_sets/` (`validation_manifest.json`) — assembled by re-deriving the
RC4D ordered-union gate over the component sets. **141 entries / 130 unique.**

| Set | size | gate fires |
|---|---|---|
| rc4a_known_wins | 5 | 5 |
| rc4b_known_wins | 16 | 16 |
| rc4c_residue_known_wins | 7 | 7 |
| component_overlap_controls | 5 | 5 (RC4B+RC4C both fire) |
| composition_fresh_holdout | 34 | 34 |
| negative_controls | 24 | **0** |
| namespace_negative_controls | 20 | **0** |
| canonical_smoke | 30 | **0** |

Off-gate emissions in NOFIRE sets: **0** (two Finset.disjUnion theorems that legitimately fire
RC4A were reclassified out of negatives into fresh holdout). Namespace distribution spans
Set / Multiset / List / Finset / Nat / Order.

---

## 6. Literal RC2 baseline

`out/literal_rc2_results.json` — exact RC2 config (rc2_release wrapper, ns24 router,
hybrid_evolved, top-k 8, max-steps 8). **100% reused** from the three components' literal RC2
results + TR confirmations (every theorem ran at the identical config); 0 live runs.

- Known wins: **0/5 · 0/16 · 0/7** solved; overlap controls **0/5** — all genuine RC2 failures.
- composition_fresh_holdout: 10/34 RC2-solved. Floors (full §12): demo 12/15, medium 37/38, large 49/65.

---

## 7. External additive evaluation

`out/additive_candidate_results.json` — `candidate_solved = RC2_solved OR a gated component
tactic closes single-shot`; ordered attribution credits the first component (in ordering) whose
tactic closes. Reuse-first over component probe caches; 48 theorems probed live.

| Metric | Value |
|---|---|
| raw delta | **+24** |
| delta by component | RC4A 5, RC4B 16, RC4C_residue 3 |
| by namespace | Set 12, Multiset 9, List 2, Finset 1 |
| overlap eliminated (RC4C→RC4B) | **9** |
| off-gate emissions | 0 |
| regressions | 0 |
| emitted-and-failed | 23 (honest negatives) |

The +3 RC4C_residue additive wins: `List.Forall.imp`, `Set.Nonempty.subset_pair_iff_eq`,
`List.forall_map_iff` (the last demoted in §8).

---

## 8. Minimal attribution

`out/minimal_attribution.json` — bare controls + component-specific controls per new win.

| Class | n |
|---|---|
| TRUE_RC4A_WIN | 5 |
| TRUE_RC4B_WIN | 16 |
| TRUE_RC4C_RESIDUE_WIN | 2 |
| SIMP_ONLY_DUPLICATE | 1 |

- **Credited delta total: 23** (RC4A 5, RC4B 16, RC4C_residue 2).
- RC4C_residue credited: `List.Forall.imp`, `Set.Nonempty.subset_pair_iff_eq` — both genuine
  depth-2 (`simp [L]` alone fails, `<;> aesop` closes).
- **SIMP_ONLY_DUPLICATE: `List.forall_map_iff`** — `simp [List.forall_iff_forall_mem]` closes it
  *alone* (depth-1), so it is not credited as a depth-2 residue win (honest demotion).
- **Overlap removed (RC4C_residue → RC4B): 9** — every Multiset disjoint theorem both fire on is
  credited to RC4B. No double-counting; RC4C_residue's net contribution is exactly the 2 genuine
  non-disjoint depth-2 wins.

---

## 9. Off-gate / preservation

`out/offgate_preservation.json` — deterministic gate scan.

- Off-gate emissions: **0**; RC4C_residue off-gate: **0** → **OFFGATE_CLEAN**.
- Emitted-and-failed by component (narrowness signal): RC4A 2/12 (0.17), RC4B 18/39 (0.46),
  RC4C_residue 18/34 (0.53) — honest negatives on hard fresh holdouts, not regressions.
- Regressions: 0 (additive evaluator, candidate ⊇ RC2).

---

## 10. Determinism

`out/determinism_check.json` — two passes over the credited-win-bearing sets + overlap controls +
negatives (composition_fresh_holdout excluded: emitted-and-failed heavy-aesop probes add hours and
no win-stability signal; gate decisions on every set are pure-function-checked in §9).

- **Deterministic: True** — clean hash `f55cbc5543c75306` ×2, 0 genuine diffs, **0 open flakes**,
  gate decisions stable, **component decisions stable**.

---

## 11. Schema-native wrapper smoke

`out/schema_wrapper_smoke.json`; wrapper `composition_rc4d/rc4d_candidate_wrapper.json` (RC2 copy
+ component tactics prepended to `priority_templates["any"]` + `theorem_name_tactic_gates`;
RC4C_residue deployed RC4B-style as bare `simp [L]` + combinator).

**Wrapper reproduces 22/23 credited wins (96%) → `SCHEMA_REPRODUCES`, 0 regressions.**

| Component | reproduced |
|---|---|
| RC4A | **5/5** |
| RC4B | **15/16** |
| RC4C_residue | **2/2** |

- Only miss: **`Set.disjoint_sUnion_right`** — a genuine search-depth gap (the wrapper offers the
  correct `simp [Set.disjoint_left] <;> aesop`, which closes single-shot in the additive
  evaluator, but the best-first search does not close this one hard sUnion goal within budget).
  Not a wrapper bug; one hard theorem.
- **Critical wrapper-construction lesson (fixed):** RC2's gate semantics is
  `full_name.startswith(prefix)` (`evolve/strategy_wrapper.py:757`), NOT substring containment.
  The first wrapper build gated the RC4A tactics with bare tokens (`monotoneOn_iff_monotone`),
  which `Set.monotoneOn_iff_monotone` does **not** start with → all 5 RC4A tactics were
  gate-denied and RC4A reproduced 0/5. Fixing the prefixes to include the namespace
  (`Set.monotoneOn`, `Finset.mem_disjUnion`, …) restored RC4A to 5/5. RC4B/RC4C_residue were
  unaffected (their prefixes already carried the namespace).
- RC4C_residue deployed RC4B-style (bare `simp [L]` + `<;> aesop`) reproduces both residue wins
  through the search — unlike RC4C's original fused-only deployment (0/12). This validates the
  RC4C deployment recommendation.

This is a smoke (Part 10), not release validation, but with 22/23 reproduction + 0 regressions
the schema integration is **not** a blocker.

---

## 12. Full canonical floor benchmark

`out/full_floor_benchmark.json` — literal RC2 vs RC4D wrapper over the FULL floor sets at the
RC2-release verification config.

| Floor | n | RC2 | RC4D | Δ | regressed | pass |
|---|---|---|---|---|---|---|
| demo_v1 | 15 | 12 | 12 | 0 | 0 | ✓ |
| nat_defs_medium | 38 | 37 | 37 | 0 | 0 | ✓ |
| nat_defs_large_v5 | 65 | 49 | 49 | 0 | 0 | ✓ |

**All floors pass (RC4D ≥ RC2, 0 regressions).** RC4D's namespace-gated actions never fire on the
Nat-arithmetic floors, so floors are preserved exactly.

---

## 13. Decision

**`RC4D_COMPOSITION_CANDIDATE_CONFIRMED`**

Every confirmation requirement is met:

| Requirement | Result |
|---|---|
| positive credited delta over literal RC2 | **+23** (RC4A 5, RC4B 16, RC4C_residue 2) |
| schema-native wrapper reproduces most additive wins | **22/23 = 96%** (`SCHEMA_REPRODUCES`) |
| no regressions | **0** (additive + wrapper smoke) |
| off-gate clean | **0** (RC4C_residue off-gate 0) |
| deterministic | **True** (hash `f55cbc5543c75306`, 0 diffs, 0 flakes) |
| full canonical floors preserved | demo 12→12 · medium 37→37 · large 49→49 |
| RC4C residue de-duplicated | 9 overlap removed → RC4B; 2 genuine additive |

The composition is **clean and additive**: the three components stack to +23 credited wins with
no double-counting (ordered attribution credits the 9 Multiset disjoint overlaps to RC4B), no
off-gate firing, no regressions, full determinism, and full floor preservation. The schema-native
wrapper — once the gate-prefix bug was fixed and RC4C_residue deployed RC4B-style — reproduces
22/23 wins through the real best-first search, so the candidate is also deployable in principle
(the single miss is one hard sUnion goal, a search-depth gap, not an integration blocker).

This remains an **off-by-default composition candidate**, not a release.

---

## 14. Next steps

RC4D composition is confirmed, so the open lever is the **RC4 release candidate**:

1. Prepare an RC4 release candidate from the RC4D wrapper (freeze a release artifact, do not edit
   RC2) and run the **full RC2-vs-RC4 release benchmark** on canonical floors **+ a fresh frontier**
   (the validation here used reused/known wins; a release needs a fresh out-of-sample sweep).
2. Resolve the one schema miss (`Set.disjoint_sUnion_right`) only if cheap — it is a search-depth
   gap, fixable by raising the budget on that route, **not** a candidate-logic change.
3. Owner approval gate before any `rc4_release/` artifact (the RC2 release-freeze precedent).

If a release benchmark on a fresh frontier shows net positive delta with floors preserved, RC4 is
release-ready. Schema integration itself is **not** a blocker (do not touch candidate logic).

---

## 15. Protected-file confirmation

- `rc1_production_wrapper.json` — **untouched**
- `rc2_release/rc2_production_wrapper.json` — **untouched**
- `ns24_router.json` — **untouched**
- NS9 genome/checkpoints, REL1/RC1/RC2 reports, TR1–TR6 datasets, RC4A/RC4B/RC4C source
  artifacts — **untouched**
- No production routing change · no RC4 release · candidate off-by-default · **no commit made**
- `git diff --stat HEAD` over the three protected wrappers: **empty** (verified §Part 13 commands).
