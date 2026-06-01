# RC4C — d2_simp_aesop Candidate Validation Report

**Candidate:** `RC2 ⊕ narrow d2_simp_aesop` (depth-2 `simp [L] <;> aesop` over a 6-lemma allowlist)
**Methodology:** RC4A/RC4B external-additive validation (literal RC2 reuse-first, single-shot
gated probe, minimal attribution, off-gate scan, determinism, schema smoke).
**Decision:** **`RC4C_CONFIRMED_WITH_RC4B_OVERLAP`**
**Status:** evidentiary only — off-by-default, NOT promoted, NO RC4 release, NO commit, protected
RC1/RC2/NS24 untouched.

---

## 1. Executive summary

RC4C survives validation as a **qualified** candidate. The pure non-overlap delta over
literal RC2 is **+7 genuine depth-2 wins** (minimal-attribution-confirmed `simp [L] <;> aesop`
where `simp [L]` alone and all bare controls fail), of which **3 are fresh out-of-sample**.
However a near-equal **8 of the candidate's wins overlap RC4B** (their lemma is
`Set/Multiset.disjoint_left`, already validated as RC4B bridge actions), and the original
evidence is overlap-dominated (8 of 12 known wins). The candidate is therefore confirmed but
its overlap with RC4B is explicitly measured and excluded from the pure credit.

| metric | value |
|---|---|
| literal RC2 on known wins | **0 / 12** (all confirmed failures) |
| candidate raw delta (all actions) | **+19** |
| candidate raw delta (nonoverlap actions) | **+11** |
| **pure RC4C credit (TRUE_D2_SIMP_AESOP_WIN)** | **7** (3 fresh / 4 reproduction) |
| overlap-RC4B wins (TRUE_D2…_OVERLAP_RC4B) | 8 |
| demoted SIMP_ONLY_DUPLICATE (depth-1, not depth-2) | 4 |
| composition-credited (pure + overlap) | 15 |
| off-gate emissions (all / nonoverlap) | **0 / 0** |
| regressions | **0** (additive evaluator) |
| determinism | **True** (clean hash `620a7e9d5dcdf044` ×2, 0 genuine diffs) |
| schema-native wrapper smoke | 0/12 via fused combinator (see §11 — deployment note) |

**Decision rationale:** pure non-overlap delta is positive (7), genuine depth-2, includes
fresh wins, 0 off-gate, 0 regressions, deterministic, narrow gates — but it is *not* a
majority of the candidate's wins (overlap 8 ≥ pure 7) and the evidence was overlap-dominated,
so the honest call is `RC4C_CONFIRMED_WITH_RC4B_OVERLAP`, not a clean standalone confirmation.

---

## 2. Background

TR6's fresh-frontier sweep surfaced **9 fresh d2_simp_aesop wins** and flagged
`RC4C_READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT`, but its own evidence file recorded
`overlap_with_rc4b` of 6/9 and a *medium* source-specific risk: "the credit is the `simp[L]`
enabling step." d2_simp_aesop is **riskier than RC4B** for three reasons:

1. **Overlap.** Two of the six allowlist lemmas (`Set.disjoint_left`, `Multiset.disjoint_left`)
   make `simp [L] <;> aesop` *literally an RC4B action*. Counting those again would double-count.
2. **Depth-2 vs depth-1.** A `simp [L] <;> aesop` "win" is only genuinely depth-2 if `simp [L]`
   alone does **not** already close the goal. If it does, the win belongs to RC4A/d1, not RC4C.
3. **`aesop` over-attribution.** `aesop` after `simp [L]` can close goals where bare `aesop`
   times out, so the marginal credit must be pinned to the named-lemma enabling step.

This validation addresses all three head-on (overlap split, `simp [L]`-alone control, bare-aesop
control).

---

## 3. Candidate definition

Six allowlisted lemmas, each emitting the single depth-2 tactic `simp [L] <;> aesop` under its
own narrow namespace + name/goal gate (`max 1 emission/theorem`):

| action | tactic | namespace | gate tokens | overlap |
|---|---|---|---|---|
| SET_DISJOINT_LEFT_D2 | `simp [Set.disjoint_left] <;> aesop` | Set | disjoint/Disjoint | **RC4B** |
| MULTISET_DISJOINT_LEFT_D2 | `simp [Multiset.disjoint_left] <;> aesop` | Multiset | disjoint/Disjoint | **RC4B** |
| MULTISET_DISJOINT_RIGHT_D2 | `simp [Multiset.disjoint_right] <;> aesop` | Multiset | disjoint/Disjoint | none |
| SET_SUBSET_PAIR_D2 | `simp [Set.subset_pair_iff_eq] <;> aesop` | Set | subset_pair/pair | none |
| FINSET_BIUNION_SUBSET_D2 | `simp [Finset.biUnion_subset] <;> aesop` | Finset | biunion | none |
| LIST_FORALL_D2 | `simp [List.forall_iff_forall_mem] <;> aesop` | List | forall | none |

Two evaluation modes: **RC4C_all** (all 6) and **RC4C_nonoverlap** (the 4 non-`disjoint_left`
actions). Excluded by construction: bare `simp <;> aesop` / `simp_all <;> aesop`, arbitrary
retrieved lemmas, broad depth-2 search, the depth-1 `simp [L]` form, and the RC4A allowlist.
The candidate is narrow (an explicit 6-lemma allowlist with per-lemma gates) — **not**
`CANDIDATE_TOO_HETEROGENEOUS`.

---

## 4. Evidence extraction (TR3 / TR5 / TR6)

`scripts/rc4c_extract_d2_simp_aesop_evidence.py` → **12 deduplicated known wins**:

- **by lemma:** `Set.disjoint_left` ×6, `Multiset.disjoint_left` ×2, `Multiset.disjoint_right` ×1,
  `Set.subset_pair_iff_eq` ×1, `Finset.biUnion_subset` ×1, `List.forall_iff_forall_mem` ×1.
- **by namespace:** Set 7, Multiset 3, Finset 1, List 1.
- **overlap:** pure non-overlap **4**, overlap-RC4B **8**, overlap-RC4A **0**. `overlap_dominates = True`.
- **fresh vs reproduction:** 9 TR6 fresh, 3 TR3/TR5 reproduction. `needs_review = 0`.

Pure-evidence bucket (A): `Set.Nonempty.subset_pair_iff_eq`, `Finset.biUnion_subset_iff_forall_subset`,
`List.Forall.imp`, `Multiset.disjoint_add_right`. The remaining 8 are the disjoint_left family,
already RC4B.

---

## 5. Validation theorem sets

`scripts/rc4c_build_validation_sets.py` → **7 sets, 149 entries / 109 unique** (manifest):

| set | n | gate fires (all) | gate fires (nonoverlap) |
|---|---|---|---|
| known_wins_all | 12 | 12 | 6 |
| known_wins_nonoverlap | 4 | 4 | 4 |
| fresh_holdout_all | 30 | 30 | 29 |
| fresh_holdout_nonoverlap | 20 | 20 | 20 |
| negative_controls | 18 | **0** | **0** |
| namespace_negative_controls | 20 | **0** | **0** |
| canonical_smoke | 45 (30 uniq) | **0** | **0** |

Gate fires 0× on every NOFIRE set in both modes (narrowness confirmed at build time).

---

## 6. Literal RC2 baseline

`scripts/rc4c_run_literal_rc2.py` (reuse-first: 65/109 reused from TR3/TR5/TR6/SF4/TR2 at the
identical config; 44 run live). Identical RC2 config: `rc2_release` wrapper, ns24 router,
`hybrid_evolved`, top-k 8, max-steps 8.

- **Known wins: 0/12 solved** (all confirmed RC2 failures — the delta is real, not stale).
- Status histogram: failed 69 / solved 38 / open_flake 2.
- Canonical floors: **demo_v1 12/15 · nat_defs_medium 14/15 · nat_defs_large_v5 14/15**.

---

## 7. Candidate evaluation

`scripts/rc4c_run_candidate_eval.py` — external additive, dual-mode
(`candidate_solved = RC2_solved OR gated simp[L]<;>aesop closes single-shot`):

| metric | all | nonoverlap |
|---|---|---|
| new wins over RC2 | **+19** | **+11** |
| by namespace | Set 7, Multiset 9, List 2, Finset 1 | Set 1, Multiset 7, List 2, Finset 1 |
| overlap-RC4B-only wins | 8 | — |
| regressions | 0 | 0 |
| off-gate emissions | 0 | 0 |
| gate emissions | 42 | — |
| emitted-and-solved / emitted-and-failed | 19 / 19 | — |

The nonoverlap delta (11) **exceeds** the 4 known nonoverlap evidence theorems — the
`Multiset.disjoint_right` action (non-overlap) independently recovers several Multiset disjoint
goals that RC4B closes via `disjoint_left`, plus fresh List/Multiset wins. The 19 emitted-and-
failed are honest negatives (the gate fires on disjoint/pair/biUnion/forall goals it cannot close
single-shot), showing the candidate is targeted, not a catch-all.

---

## 8. Minimal attribution (authoritative)

`scripts/rc4c_minimal_attribution.py` re-probes every new win with bare controls (`simp`,
`simp_all`, `aesop`, `classical <;> aesop`), and per allowlist lemma the depth-1 `simp [L]`,
the depth-2 `simp [L] <;> aesop`, and lemma-direct (`exact`/`simpa using`). A win is genuine
depth-2 only if `simp [L] <;> aesop` closes **and `simp [L]` alone does not**.

| classification | n | targets |
|---|---|---|
| **TRUE_D2_SIMP_AESOP_WIN** (pure RC4C credit) | **7** | List.Forall.imp · Multiset.disjoint_add_left · Multiset.disjoint_add_right · Set.Nonempty.subset_pair_iff_eq · Multiset.disjoint_iff_ne · Multiset.disjoint_union_left · Multiset.singleton_disjoint |
| TRUE_D2_SIMP_AESOP_OVERLAP_RC4B | 8 | Set.disjoint_{iUnion,sUnion}_{left,right} · Set.disjoint_right · Set.disjoint_iff_forall_ne · Multiset.disjoint_singleton · Multiset.disjoint_right |
| SIMP_ONLY_DUPLICATE (depth-1, demoted) | 4 | **Finset.biUnion_subset_iff_forall_subset** · List.forall_map_iff · Multiset.disjoint_cons_left · Multiset.zero_disjoint |

- **Pure RC4C by namespace:** Multiset 5, List 1, Set 1.
- **Pure fresh (out-of-sample):** Multiset.disjoint_iff_ne, Multiset.disjoint_union_left,
  Multiset.singleton_disjoint (all via the non-overlap `simp [Multiset.disjoint_right] <;> aesop`).
- **Pure reproduction:** List.Forall.imp, Multiset.disjoint_add_left, Multiset.disjoint_add_right,
  Set.Nonempty.subset_pair_iff_eq.
- **Honest demotion:** `Finset.biUnion_subset_iff_forall_subset` — one of the four headline
  pure-evidence theorems — is **SIMP_ONLY_DUPLICATE**: `simp [Finset.biUnion_subset]` closes it
  *alone*, so it is depth-1 (RC4A/d1), not a genuine RC4C depth-2 win. The attribution caught
  this; it is not credited to RC4C.

Pure RC4C delta = **7**; composition-credited (pure + RC4B-overlap) = **15**.

---

## 9. Off-gate / preservation

`scripts/rc4c_offgate_preservation.py` → **`OFFGATE_CLEAN`**:

- off-gate emissions: **0 (all) / 0 (nonoverlap)** — 0 fires on negative_controls,
  namespace_negative_controls, canonical_smoke in both modes.
- regressions: 0 (additive evaluator, candidate ⊇ RC2 — structurally impossible).
- emitted-and-failed: 19/42 (rate 0.45) — honest negatives, not regressions; expected for a
  targeted bridge over hard disjoint/pair/biUnion goals.
- canonical floors (literal RC2): demo_v1 12/15 · medium 14/15 · large 14/15, 0 gate fires.

No broad-gate warning. Not `REJECT_BROAD_GATE`.

---

## 10. Determinism

`scripts/rc4c_determinism_check.py` (two passes over known_wins + fresh_holdout_all + negative
controls; hash over cleanly-executed theorems, infra flakes excluded — RC4B methodology):

- **deterministic = True**; clean run1 hash `620a7e9d5dcdf044` = run2 hash; **0 genuine diffs**;
  gate decisions stable across all targets.
- 4 open flakes (Dojo hard-timeout / worker-kill on heavy `<;> aesop` hard goals:
  List.filterMap_eq_map_iff_forall_eq_some, Multiset.singleton_disjoint, Multiset.zero_disjoint,
  Set._root_.Disjoint.image).
- 2 win-affecting flakes: **Multiset.singleton_disjoint** (a credited pure win — solves when the
  probe completes within budget, worker-killed otherwise) and Multiset.zero_disjoint (already
  demoted to SIMP_ONLY_DUPLICATE, not credited). Same infrastructure-flake class RC4B documented;
  excluded from the hash, flagged here for honesty.

---

## 11. Schema-native wrapper smoke (Part 10, optional — NOT release validation)

`scripts/rc4c_schema_wrapper_smoke.py` builds `rc4c_candidate_wrapper.json` (RC2 copy + the six
`simp [L] <;> aesop` tactics prepended to `priority_templates["any"]` + `theorem_name_tactic_gates`)
and runs it through the real `eval_rollout_all` search over known_wins_all + the negative controls.

- **known wins reproduced: 0/12**, **regressions: 0**, broad perturbation: 0 (negative controls
  unchanged; the 8 wrapper-solved smoke theorems are all RC2-solvable controls).
- **Diagnosis (verified):** the wrapper is well-formed (tactics prepended + gated) and the
  combinator solves **single-shot** through `env.run_transition` (external evaluator: success), but
  the best-first **search** does not reproduce it via the *fused* `simp [L] <;> aesop` priority
  template. This contrasts with RC4B's smoke (10/11), which also prepended the **bare depth-1**
  `simp [L]` — there the search chained `simp [L]` (advance) → RC2's own `aesop` (close). RC4C added
  only the fused combinator, which the harness's priority-template search applies differently.
- **Conclusion:** this is a **deployment-integration note**, not a validation failure. The external
  additive evaluator remains the authority (+7 pure confirmed single-shot). Recommended deployment
  for any future RC4 composition: integrate RC4C lemmas as the **bare `simp [L]` enabling action**
  (RC4B-style, letting the search's `aesop` close) rather than the fused combinator, and re-smoke.

---

## 12. Decision

### `RC4C_CONFIRMED_WITH_RC4B_OVERLAP`

Allowed-criteria met:
- **Total candidate useful:** +19 raw / 15 composition-credited wins.
- **Overlap clearly measured:** 8 wins overlap RC4B (`disjoint_left` lemmas), excluded from pure credit.
- **Pure non-overlap nonzero & confirmed:** +7 genuine depth-2 wins (3 fresh), minimal-attribution
  verified (`simp [L]` alone fails), 0 off-gate, 0 regressions, deterministic, narrow gates.

Why not the stronger `RC4C_CANDIDATE_CONFIRMED`: the pure non-overlap delta (7), while positive and
genuine, is **not a majority** of the candidate's wins (overlap 8 ≥ pure 7) and the evidence was
overlap-dominated; one headline pure-evidence lemma (`Finset.biUnion_subset`) demoted to depth-1.
Honesty favors the qualified verdict. Why not a REJECT: literal delta is positive and attributed
(not `REJECT_NO_LITERAL_DELTA`), gates are narrow with 0 off-gate (not `REJECT_BROAD_GATE`), 0
regressions (not `REJECT_REGRESSION`), and the family is a coherent 6-lemma allowlist (not
`CANDIDATE_TOO_HETEROGENEOUS`).

---

## 13. Next steps

1. **RC4 composition.** RC4C's pure contribution (7) is meaningful but largely Multiset-disjoint-
   shaped and partly recovers what RC4B already covers. Before an RC4 composition
   (`RC2 ⊕ RC4A ⊕ RC4B ⊕ RC4C`), de-duplicate RC4C against RC4B: the genuinely additive RC4C
   material is `Multiset.disjoint_right` (Multiset disjoint), `Set.subset_pair_iff_eq`, and
   `List.forall_iff_forall_mem` (drop `Finset.biUnion_subset` → it's depth-1/RC4A-class).
2. **Deployment form.** Integrate RC4C as the bare `simp [L]` enabling action + search-`aesop`
   (RC4B-style), re-run the schema smoke, and benchmark the composition with literal floors before
   any release.
3. **Overlap consolidation.** Because 8/15 composition wins are RC4B, consider keeping the
   `disjoint_left` depth-2 wins under RC4B only and shipping RC4C as the 3-lemma non-disjoint_left
   residue.

---

## 14. Protected-file confirmation

- `project/evolve/experiments/rc1/rc1_production_wrapper.json` — **untouched**.
- `project/evolve/experiments/rc2_release/rc2_production_wrapper.json` — **untouched**.
- `project/evolve/routing/ns24_router.json` — **untouched**.
- NS9 genome/checkpoint, REL1/RC1/RC2 reports, TR1–TR6 datasets, RC4A & RC4B artifacts — **untouched**.
- No production routing change · no RC4 release · candidate off-by-default · **no commit made**.

All new files are under `project/evolve/experiments/rc4_candidates/d2_simp_aesop/`,
`project/evolve/reports/rc4/`, and `scripts/rc4c_*.py`.
