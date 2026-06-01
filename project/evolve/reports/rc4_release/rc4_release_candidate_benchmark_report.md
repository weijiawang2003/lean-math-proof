# RC4 Release Candidate Benchmark Report (RC4R)

**Candidate:** RC4 = RC2 ⊕ RC4A ⊕ RC4B ⊕ RC4C_residue (= validated RC4D composition)
**Status:** off-by-default **release candidate** — NOT a production release; RC2 stays production.
**Date:** 2026-06-01

---

## 1. Executive summary

| Metric | Result |
|---|---|
| RC2 benchmark solved | 78 / 271 |
| RC4 benchmark solved | 100 / 271 |
| Raw delta | +22 |
| New wins / regressions / **net delta** | 22 / 0 / **+22** |
| New wins by component | RC4A 5, RC4B 15, RC4C_residue 2 |
| Known-win reproductions / fresh new wins | 22 / 0 |
| Off-gate audit | OFFGATE_CLEAN (0) |
| Full floor verification | all pass (12/37/49, 0 regr) |
| Determinism rerun | deterministic (hash ded0e256, 0 diffs) |
| Fresh frontier verdict | NO_FRESH_DELTA_BUT_SAFE |
| Wrapper diff | WRAPPER_DIFF_CLEAN (15 actions added, RC2 preserved) |

**Decision: RC4_SAFE_BUT_NO_FRESH_DELTA** (see §12).

---

## 2. Background

- **RC2** = RC1 ⊕ SET_ITE_SIMP — frozen production stack (release floors demo 12 / medium 37 / large 49).
- **RC4A** def_unfold_simp — CONFIRMED (+5). **RC4B** disjoint_left bridge — CONFIRMED (+16, 5 fresh).
  **RC4C** d2_simp_aesop — CONFIRMED_WITH_RC4B_OVERLAP (de-dup'd into RC4C_residue = 3 non-overlap lemmas).
- **RC4D** composition — `RC4D_COMPOSITION_CANDIDATE_CONFIRMED`: +23 credited, 9 overlaps de-dup'd
  to RC4B, 0 off-gate, 0 regressions, deterministic, floors preserved, schema wrapper 22/23.
- RC4R is the release-candidate benchmark: a clean RC2-based wrapper + formal RC2-vs-RC4 run
  including a **fresh out-of-sample frontier** (the measurement RC4D's reused/known-win validation lacked).

---

## 3. RC4 wrapper construction

`rc4_release_candidate_wrapper.json` — `out/rc4_wrapper_diff.json`: **WRAPPER_DIFF_CLEAN**.

- Clean copy of RC2 + 15 validated RC4D tactics prepended to `priority_templates["any"]` + 15
  name-prefix gates. RC2 fields preserved exactly; 0 unrelated changes.
- Component mapping: RC4A 5 tactics, RC4B 4, RC4C_residue 6.
- **Purely additive**: `theorem_name_tactic_gates` match `full_name.startswith(prefix)`; a theorem
  matching no prefix is byte-identical to RC2 ⇒ RC4 ≡ RC2 on all non-gate-firing theorems.

---

## 4. Benchmark theorem sets

`theorem_sets/benchmark_manifest.json` — 355 entries / 271 unique.

| set | size | gate fires | fresh |
|---|---|---|---|
| canonical_demo_v1 | 15 | 0 | reused |
| canonical_nat_defs_medium | 38 | 0 | reused |
| canonical_nat_defs_large_v5 | 65 | 0 | reused |
| rc4_known_wins | 23 | 23 | known |
| fresh_out_of_sample_frontier | 125 | 80 | fresh |
| negative_controls | 44 | 0 | — |
| offgate_controls | 45 | 0 | — |

Fresh frontier component split (gate firing): RC4A 71, RC4C_residue 9, RC4B 2; balanced across
Set/Multiset/Finset/List/Nat/Order/Other.

---

## 5. RC2 baseline benchmark

`out/rc2_benchmark_results.json` — exact RC2 config; floors + known-wins + controls reused from
RC4D, fresh frontier run live.

RC2 solved **78/271** (151 reused at exact config, 120 fresh run live).

| set | n | RC2 solved |
|---|---|---|
| canonical_demo_v1 | 15 | 12 |
| canonical_nat_defs_medium | 38 | 37 |
| canonical_nat_defs_large_v5 | 65 | 49 |
| rc4_known_wins | 23 | 0 |
| fresh_out_of_sample_frontier | 125 | 11 |
| negative_controls | 44 | 6 |
| offgate_controls | 45 | 7 |

Status: solved 78 / failed 181 / open_flake 12.

---

## 6. RC4 benchmark

`out/rc4_benchmark_results.json` — RC4 release wrapper; floors + known-wins reused from RC4D
(identical wrapper), non-gate-firing theorems forced RC4 ≡ RC2 (additive), fresh gate-firing run live.

RC4 solved **100/271** (floors + 23 known-wins reused from RC4D; 168 non-gate-firing
forced RC4 ≡ RC2; 80 fresh gate-firing run live).

| set | n | RC4 solved |
|---|---|---|
| canonical_demo_v1 | 15 | 12 |
| canonical_nat_defs_medium | 38 | 37 |
| canonical_nat_defs_large_v5 | 65 | 49 |
| rc4_known_wins | 23 | **22** |
| fresh_out_of_sample_frontier | 125 | 11 |
| negative_controls | 44 | 6 |
| offgate_controls | 45 | 7 |

Status: solved 100 / failed 159 / open_flake 12. The single known-win miss is
`Set.disjoint_sUnion_right` (the RC4D schema-smoke search-depth gap on one hard sUnion goal).

---

## 7. RC4 vs RC2 comparison

`out/rc4_vs_rc2_comparison.json`.

| Metric | Value |
|---|---|
| RC2 solved | 78 |
| RC4 solved | 100 |
| raw delta | **+22** |
| RC4 new wins | **22** |
| RC4 regressions | **0** |
| **net delta** | **+22** |
| by component | RC4A 5, RC4B 15, RC4C_residue 2 |
| known-win reproductions | 22 |
| **fresh out-of-sample new wins** | **0** |

The entire +22 is **known-win reproduction**: RC4 solves 22 of the 23 RC4D-attributed theorems
that RC2 cannot, across RC4A (Set monotone/antitone def-unfold), RC4B (Set/Multiset disjoint
bridge), and RC4C_residue (Multiset disjoint_right / List forall depth-2). **0 regressions.**

---

## 8. Off-gate audit

`out/rc4_offgate_audit.json`.

**OFFGATE_CLEAN — 0 off-gate emissions** on negative_controls + offgate_controls (the gate is a
pure function; verified over all 271 entries).

Component emissions (gate-firing theorems): RC4A 76, RC4B 18, RC4C_residue 20.
Emitted-and-failed rate (narrowness signal, honest negatives — not regressions):

| component | fired | failed | rate |
|---|---|---|---|
| RC4A | 76 | 69 | **0.91** ⚠ |
| RC4B | 18 | 3 | 0.17 |
| RC4C_residue | 20 | 8 | 0.40 |

**Broad-gate warning (RC4A):** RC4A's def-unfold gate fires on 76 monotone/antitone/disjUnion
theorems but closes only ~7 — on fresh mono theorems the `simp [Monotone,…]` unfold rarely
finishes the proof. This is on-gate (the def IS in the goal) and harmless (additive, no
regression, no off-gate), but it shows RC4A's gate is **broad relative to its win rate**. RC4B
and RC4C_residue are tight.

---

## 9. Full floor verification

`out/rc4_full_floor_verification.json`.

**All canonical floors pass — 0 regressions.**

| floor | n | RC2 | RC4 | release ref | pass |
|---|---|---|---|---|---|
| demo_v1 | 15 | 12 | 12 | 12 | ✓ |
| nat_defs_medium | 38 | 37 | 37 | 37 | ✓ |
| nat_defs_large_v5 | 65 | 49 | 49 | 49 | ✓ |

RC4 ≥ RC2 ≥ RC2-release-floor on every floor; the namespace-gated RC4 actions never fire on the
Nat-arithmetic floors, so floors are preserved exactly.

---

## 10. Determinism rerun

`out/rc4_determinism_rerun.json`.

**Deterministic: True.** Reran the RC4 wrapper through the search twice over 63 targets
(23 known wins + demo_v1 floor + 25 gate-firing fresh): clean hash `ded0e256b75a12eb` ×2,
**0 genuine diffs**, 2 open flakes (Dojo hard-timeout / worker-kill infra events, not
win-affecting). RC4's wins/regressions are stable across reruns.

---

## 11. Fresh frontier analysis

`out/rc4_fresh_frontier_analysis.json`.

**Verdict: `NO_FRESH_DELTA_BUT_SAFE`.**

| Metric | Value |
|---|---|
| fresh theorems | 125 (113 analyzable, 12 flakes) |
| RC2 solved | 11 |
| RC4 solved | 11 |
| **fresh new wins** | **0** |
| fresh regressions | **0** |

On 80 genuinely fresh out-of-sample gate-firing theorems (RC4A 71, RC4C_residue 9, RC4B 2 —
none used in RC4D validation), RC4's gates fire but close **0 new** theorems beyond RC2, and
cause **0 regressions**. RC4 does **not** improve beyond known-win replay on fresh material —
consistent with the TR5/RC4D caveat that the components reproduce their evidence wins but the
fresh out-of-sample yield is near zero (the fresh RC4A mono theorems are not simple iff-unfolds;
only 2 fresh RC4B and 9 fresh RC4C_residue candidates existed, none closing). RC4 is **safe** on
fresh theorems (additive, no regressions) but adds no fresh coverage.

---

## 12. Decision

**RC4_SAFE_BUT_NO_FRESH_DELTA**

RC4 passes **every safety gate**: net **+22** over RC2 (RC4A 5, RC4B 15, RC4C_residue 2),
**0 regressions**, **0 off-gate**, all canonical floors preserved (12/37/49), **clean wrapper
diff**, and **deterministic**. By the literal criteria these meet
`RC4_RELEASE_CANDIDATE_RECOMMENDED` (which requires only "at least known-win reproduction,
preferably fresh frontier delta").

The honest verdict is the more precise **`RC4_SAFE_BUT_NO_FRESH_DELTA`**, because:

1. **The entire +22 is known-win reproduction.** On 80 genuinely fresh out-of-sample gate-firing
   theorems (none used in RC4D validation), RC4 adds **0 new wins** and **0 regressions**. RC4
   does exactly what it was built for and no more — there is no evidence it generalizes to novel
   theorems.
2. **RC4A's gate is broad relative to its win rate** — it fires on 76 fresh monotone/antitone/
   disjUnion theorems and closes only ~7 (0.91 emitted-and-failed). On-gate and harmless
   (additive, no regression), but it is the loosest component.

**This is still a positive, deployable outcome.** The +22 are real Mathlib theorems RC2 cannot
prove and RC4 can (Set monotone/antitone lemmas, Set/Multiset disjoint lemmas, Multiset
disjoint_right / List forall) — shipping RC4 strictly improves production coverage by 22 with
zero downside. So RC4 is **owner-approvable for the +22 known gains**; it is just not a
*generalizing* improvement, and the owner should approve it as "lock in the validated component
wins," not as "RC4 finds new things." RC2 stays production until the owner decides.

---

## 13. Owner approval checklist

- **Wrapper path:** `project/evolve/experiments/rc4_release_candidate/rc4_release_candidate_wrapper.json`
- **Benchmark artifacts:** `project/evolve/experiments/rc4_release_candidate/out/` + this report.
- **Exact delta:** RC2 78 → RC4 100 solved (net +22: RC4A 5, RC4B 15, RC4C_residue 2); 0 regressions; fresh out-of-sample delta 0.
- **Known caveats:** The +22 is 100% known-win reproduction (in-sample); 0 fresh out-of-sample delta (NO_FRESH_DELTA_BUT_SAFE). One known win not reproduced through the search (`Set.disjoint_sUnion_right`, depth gap). RC4A gate fires broadly (76×) but closes few (~7).
- **Remaining risks:** RC4A def-unfold gate is broad (low win-rate on fresh mono) — additive/safe but worth tightening before any future expansion. No fresh-frontier generalization signal, so RC4 will not improve coverage beyond these component families. 12 benchmark flakes (infra, excluded from wins/regressions).
- **Recommended commit message (only if owner approves):** `Add RC4 release-candidate benchmark (RC2⊕RC4A⊕RC4B⊕RC4C_residue): +22 known wins, 0 regr/off-gate, floors preserved, no fresh delta` — only if owner approves promoting RC4 to a release artifact.

---

## 14. Protected-file confirmation

- RC1 wrapper — untouched · RC2 release wrapper — untouched · NS24 router — untouched
- NS9 checkpoints, REL1/RC1/RC2 reports, TR1–TR6 datasets, RC4A/B/C/D source artifacts — untouched
- No production routing change · no RC2 replacement · no RC4 production release · **no commit made**
- `git diff --stat HEAD` over protected wrappers + router: **empty**.
