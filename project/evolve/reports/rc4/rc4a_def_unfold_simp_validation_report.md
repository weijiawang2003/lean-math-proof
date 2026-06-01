# RC4A Candidate Validation — `def_unfold_simp`

**Decision: `RC4A_CANDIDATE_CONFIRMED`** (candidate, **not promoted**; owner approval +
full-floor + schema-native wrapper validation still required).

---

## 1. Executive summary

The narrow `def_unfold_simp` family — `simp [D, …]` where `D` is a Mathlib *definition*
(not `@[simp]`) named in the goal, restricted to a validated 9-definition allowlist —
survives literal-RC2 validation:

- **Credited delta: +5 new wins over literal RC2**, all classified
  `TRUE_DEF_UNFOLD_SIMP_WIN` by minimal attribution (bare controls fail, gated
  `simp [defs]` closes).
- **Off-gate: 0** (0/20 negative controls, 0/45 canonical smoke).
- **Regressions: 0** (structurally impossible — additive evaluator, candidate ⊇ RC2).
- **Canonical floors preserved**, 0 gate-fires on any floor theorem.
- **Deterministic: True** (run1 hash `7c6eb2db19043c83` == run2, 0 diffs, 0 flakes).
- **Narrow gate:** fires on 11/61 validation theorems, 0 on negatives/floors.

The 5 wins are the TR3 `def_unfold_simp` wins reproduced through the literal-RC2
additive harness (4 order-predicate `<pred>On_iff_<pred>` + 1 `Finset.mem_disjUnion`).
Confirmation rests on **strong repeated same-family support** (5 wins across 2
mechanistically-identical definitional sub-shapes), not on a fresh out-of-sample win —
see §6/§10 for that honest limitation.

---

## 2. Background

TR3 (Retrieval-Aware Depth Search) found 12 TRUE_DELTA over literal RC2 across 92
confirmed failures and flagged 3 `FOUND_RC_CANDIDATE_FAMILY` families, but measured no
off-gate / floors / determinism. RC4A validates the **narrowest and cleanest** of the
three — `def_unfold_simp` — first, following the SET_ITE_SIMP → RC2 methodology. The
`d2_simp_aesop` family and the `Set.disjoint_left` bridge are deferred to separate
validations (RC4B+).

## 3. Candidate definition

- **Mechanism:** for a theorem, match the allowlisted definitions whose name appears
  in the goal/statement and emit a single `simp [<matched defs>]`. Fires **only** when
  ≥1 allowlisted definition is present in the goal.
- **Validated allowlist (9):** `Monotone, MonotoneOn, Antitone, AntitoneOn, StrictMono,
  StrictMonoOn, StrictAnti, StrictAntiOn, Finset.disjUnion` — exactly the defs from the
  5 TR3 `TRUE_RETRIEVAL_ONLY_DELTA` def-unfold wins.
- **Why narrow:** allowlisted defs only; one emission/theorem; goal-presence gate
  prevents firing on Nat/arith/Multiset/List goals. NOT broad simp, NOT arbitrary
  retrieved-lemma simp, NOT `@[simp]` additions, NOT `d2_simp_aesop`, NOT the
  `Set.disjoint_left` bridge.
- **Homogeneity:** all 5 wins share one mechanism (definitional unfold); subfamilies
  (`order_predicate_def_unfold` ×4, `finset_def_unfold` ×1) differ only in *which*
  defs. → **not** `CANDIDATE_TOO_HETEROGENEOUS`.
- **Evaluation mode:** external additive — `candidate_solved = RC2_solved OR (gate
  fires AND gated simp closes single-shot)`. Avoids search-perturbation artifacts;
  makes regressions structurally impossible.

Policy: `def_unfold_simp_policy.json`.

## 4. Validation theorem sets

| set | size | gate-fires | TR3 overlap | role |
|---|---|---|---|---|
| known_wins | 5 | 5 | 5 | the TR3 def_unfold wins |
| same_cluster_holdout | 3 | 3 | 3 | gate fires, win NOT guaranteed |
| fresh_frontier_holdout | 3 | 3 | 0 | fresh Finset.disjUnion theorems |
| negative_controls | 20 | **0** | 20 | gate must not fire |
| canonical_smoke | 45 | **0** | 4 | demo_v1 + nat_defs_medium + nat_defs_large samples |

`same_cluster_holdout` = the 2 `Set.not_monotoneOn_not_antitoneOn_iff_exists_*`
confirmed failures (name MonotoneOn/AntitoneOn) + `Finset.filter_cons` (RC2-solved).
`fresh_frontier_holdout` = `Finset.coe_disjUnion`, `Finset.disjUnion_singleton`,
`Finset.disjUnion_eq_union` (0 TR3 overlap). Manifest: `theorem_sets/validation_manifest.json`.

## 5. Literal RC2 baseline

Config: `rc2_release/rc2_production_wrapper.json`, `ns24_router.json`,
`hybrid_evolved`, top-k 8, max-steps 8 — identical to SF4/TR2/TR3, so 30 cases were
reused and 31 run live. Result over 61 unique validation theorems: **31 failed / 30
solved**. Floor samples: demo_v1 12/15, nat_defs_medium 14/15, nat_defs_large 14/15.
Commands: `out/literal_rc2_commands.sh`; results: `out/literal_rc2_results.*`.

## 6. Candidate evaluation

| metric | value |
|---|---|
| theorems | 61 |
| RC2 solved | 30 |
| candidate solved | 35 |
| **new wins over RC2** | **5** |
| regressions | 0 |
| gate emissions | 11 |
| off-gate emissions | **0** |
| emitted-and-solved | 5 |
| emitted-and-failed | 6 |

**New wins (5):** `Set.monotoneOn_iff_monotone`, `Set.antitoneOn_iff_antitone`,
`Set.strictMonoOn_iff_strictMono`, `Set.strictAntiOn_iff_strictAnti`,
`Finset.mem_disjUnion`.

**Emitted-and-failed (6) — the honest generalization picture:** the gate fired but
produced no new win on the 2 `not_monotoneOn_not_antitoneOn_iff_*` same-cluster
confirmed failures (the def-unfold alone does **not** close them — they remain
PROOF_DEPTH, as TR3 found) and on the 4 RC2-already-solved disjUnion/filter cases
(additive no-ops). So the family does **not** broadly generalize even within the
order-predicate cluster; it wins precisely on the simple `<pred>On_iff_<pred>` shape
plus `Finset.mem_disjUnion`. Results: `out/candidate_results.*`.

## 7. Minimal attribution

All 5 new wins → **`TRUE_DEF_UNFOLD_SIMP_WIN`** (5/5): for each, the bare controls
`simp` / `simp_all` / `aesop` / `classical <;> aesop` all fail and the gated
`simp [defs]` closes; the unfolded defs are all in the allowlist. 0 BASELINE_DUPLICATE,
0 RC2_ALREADY_SOLVED, 0 SOURCE_SPECIFIC, 0 HETEROGENEOUS_MECHANISM. Results:
`out/minimal_attribution.*`.

## 8. Off-gate / preservation

- Off-gate emissions: **0** (negative_controls 0/20, canonical_smoke 0/45). Verdict
  `OFFGATE_CLEAN` (not `REJECT_BROAD_GATE`).
- Regressions: **0** (additive evaluator).
- Canonical floors: 0 gate-fires on any demo_v1 / nat_defs_medium / nat_defs_large
  sample; literal-RC2 floor pass rates preserved (the candidate is a strict superset).
- Results: `out/offgate_preservation.*`.

## 9. Determinism

Re-ran the 8 gate-firing probes (known_wins + same_cluster_holdout; negatives
contribute 0 probes) twice: **run1 hash `7c6eb2db19043c83` == run2 hash**, 0 diffs, 0
open flakes → **deterministic = True**. Results: `out/determinism_check.json`.

## 10. Decision

**`RC4A_CANDIDATE_CONFIRMED`.** All stated requirements are met:

| requirement | status |
|---|---|
| positive delta over literal RC2 | ✅ +5 |
| ≥1 fresh/holdout win OR strong repeated same-family support | ✅ strong repeated same-family support (5 wins, 2 subfamilies) |
| 0 off-gate | ✅ |
| 0 regressions | ✅ |
| minimal attribution confirms | ✅ 5/5 TRUE_DEF_UNFOLD_SIMP_WIN |
| deterministic | ✅ hash-stable |
| narrow gate | ✅ 11/61, 0 on negatives/floors |

**Honest caveats (do not over-read the confirmation):**
1. The 5 wins are TR3 wins reproduced in-sample through the literal-RC2 harness — there
   is **no out-of-sample NEW win**: the only non-trivial same-cluster confirmed
   failures (the 2 `not_monotoneOn` cases) did **not** yield to the def-unfold, and the
   fresh disjUnion holdouts were already RC2-solved. The family is genuinely narrow.
2. Canonical floors were **sampled** (15 each), not the full medium/large sets.
3. Validation used the external additive evaluator; a **schema-native wrapper** run is
   still needed to confirm no search-perturbation before any release.

These caveats mean *confirmed as an RC4 candidate*, not *ready to ship*.

## 11. Next steps

Confirmed path:
1. **Separately validate the `Set.disjoint_left` bridge** (RC4B) and `d2_simp_aesop`
   (RC4C) with this same harness.
2. **Full-floor + schema-native wrapper** run for `def_unfold_simp` (full
   nat_defs_medium/large, demo_v1; build `rc4a_candidate_wrapper.json` with the
   allowlist gate in `priority_templates`, per the RC2 SET_ITE lesson) before proposing
   an RC4 composition.
3. **Owner approval** required before any promotion; RC4 composition only after each
   candidate family passes independently.
4. Retain the TR3 labels + this validation for a retrieval-aware router (TR4).

## 12. Protected-file confirmation

- `rc1_production_wrapper.json` — untouched.
- `rc2_release/rc2_production_wrapper.json` — untouched (read-only strategy-config).
- `ns24_router.json` — untouched (read-only route-config).
- NS9 genome/checkpoints, REL1/RC1/RC2 reports, TR1/TR2/SF/TR datasets — untouched.
- No production routing changed; **no RC4 release created**; no README update; candidate
  **not promoted**; no commit. `git diff --stat HEAD` over the three protected wrappers
  is empty. All artifacts under `project/evolve/experiments/rc4_candidates/def_unfold_simp/`
  & `project/evolve/reports/rc4/`, scripts `scripts/rc4a_*.py`.
