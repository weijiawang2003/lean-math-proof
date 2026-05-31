# RC2 Candidate Validation — `SET_ITE_SIMP` Literal-RC1 Confirmation

Branch: `rc1-production-stack` · live LeanDojo · **no commit** · RC1/NS24/NS9 untouched.
Candidate: the narrow `SET_ITE_SIMP` gate (`simp [Set.ite]`) only, additive & off-by-default.

---

## 1. Executive summary

`SET_ITE_SIMP` **survives literal-RC1 confirmation**. Re-running the unmodified RC1
production wrapper (`rc1_production_wrapper.json` + `ns24_router.json`,
`hybrid_evolved`, top-k 8, max-steps 8) on the validation sets and then trying
`simp [Set.ite]` additively on RC1-failed gate-fired theorems:

| metric | result |
|---|---|
| literal RC1 solved | known_wins 0/5, selected 0/12, holdout 11/20 |
| candidate solved | 16/32 unique |
| **new wins over literal RC1** | **+5** (2 reproduction + **3 fresh holdout**) |
| regressions | **0** (additive; RC1 behavior unaltered) |
| off-gate emissions | **0** |
| minimal-sufficient attribution | **5/5 TRUE_SET_ITE_SIMP_WIN** (0 baseline-dup, 0 parser-artifact) |
| determinism | **deterministic** (run1 == run2 = `537e5b61ef2baf63`, 0 diffs) |

The five literal-RC1-confirmed wins: `Set.ite_empty_right`, `Set.ite_right`,
`Set.ite_empty`, `Set.ite_empty_left`, `Set.ite_left`. Each is failed by literal RC1
AND all four baselines (simp/simp_all/aesop/classical<;>aesop), and closed by
`simp [Set.ite]`.

**Recommendation: `PROMOTE_TO_RC2_CANDIDATE`** — narrow `SET_ITE_SIMP` only. All six
RC2 promotion conditions pass (see §10). Per run discipline this is a candidate, not
a direct production merge: actual RC2 composition needs project-owner approval +
canonical-floor confirmation (§11).

---

## 2. Background

SF2's Multiset-singleton and Set-cluster deep dives both returned **0 missing
lemmas** — the frontier is automation/short-sequence gaps over existing Mathlib
lemmas. SX2 mined the 10 successful Set probes into templates; only `SET_ITE_SIMP`
(`simp [Set.ite]`) was gate-worthy (theorem-agnostic, the single emittable tactic
that generalizes). It produced 5 TRUE_SET2_WIN vs a **4-tactic baseline proxy**. The
3 speculative gates (`SET_EXT_SIMP`, `SET_SUBSET_ANTISYMM`, `SET_IFF_CONSTRUCTOR`)
fired 48× for 0 true wins → disabled. This run replaces the proxy with **literal
RC1** because RC1's real battery (retrieval, priority templates, aesop/tauto) is
broader than 4 baselines.

---

## 3. Candidate definition

- **Tactic:** `simp [Set.ite]` (definitional unfold of `Set.ite` + simp close).
- **Gate (narrow):** name contains `Set` AND name/goal contains an `ite`/`if` token
  AND name/goal contains none of `Multiset`/`Nat`/`Int`; `max_emissions_per_theorem: 1`.
- **Why narrow:** it must fire only where RC1's base policy is weak (Set quotient/
  definitional ite), never on Nat/Int/Multiset surfaces.
- **Why additive / off-by-default:** evaluated as `candidate_finished =
  literal_rc1_finished OR (gate_fired AND simp[Set.ite] solves)`. RC1 is run first,
  unmodified; the candidate only ever *adds* a solve on an RC1-failure. Regressions
  are impossible by construction.

Config: `set_ite_simp_gate_policy.json` (one active gate; six gates explicitly
disabled), `set_ite_simp_candidate_wrapper.json`.

---

## 4. Validation theorem sets

`theorem_sets/` + `validation_manifest.json`:

| set | size | live | role |
|---|---|---|---|
| `set_ite_known_wins` | 5 | 5 | the 5 SX2 TRUE_SET2_WIN theorems |
| `set_ite_selected_failures` | 12 | 12 | the 12 SF2 deep-dive selected Set failures |
| `set_ite_fresh_holdout` | 20 | 20 | the 20 SX2 holdout Set theorems (paths) |
| `set_ite_negative_controls` | 5 | 1 | Nat/Int/Multiset; gate must NOT fire |
| `set_ite_canonical_smoke` | 5 | 0 | canonical non-Set preservation sample |

**Data-leakage notes:** `known_wins` ⊂ `selected` (2) ∪ `holdout` (3); `fresh_holdout`
excludes the 12 selected and the 3 SF2 deferrals (SX2 construction) but contains the
3 holdout known-wins (flagged `is_known_win`). Unique live theorems across the three
Set sets = 32. Negative/canonical sets are non-Set (gate cannot fire on them).

---

## 5. Literal RC1 baseline

Command (replayable in `out/literal_rc1_commands.sh`):
```
python3 scripts/sf1_run_eval.py --theorem-set-file <set>.json --register-name rc2_<set> \
  -- --policy-type hybrid_evolved \
     --route-config project/evolve/routing/ns24_router.json \
     --strategy-config project/evolve/experiments/rc1/rc1_production_wrapper.json \
     --top-k 8 --max-steps 8 --out-dir <out>
```
Authoritative solved flag = `per_theorem.finished` in each run's `metrics.json`.

| set | finished | elapsed |
|---|---|---|
| `set_ite_known_wins` | **0/5** | 49s |
| `set_ite_selected_failures` | **0/12** | 103s |
| `set_ite_fresh_holdout` | **11/20** | 158s |

The 11 holdout solves are all **non-ite** (`mem_dite*`, `inclusion_*`, `insert_diff_*`,
`diff_union_inter` via aesop/tauto/simp_all). **Every `Set.ite`-unfold target
(`ite_empty`, `ite_empty_left`, `ite_left`, plus `ite_compl`, `ite_inter_of_inter_eq`)
is literal-RC1 FAILED.** Note `Set.inclusion_right` — the SX2 speculative
`SET_EXT_SIMP` "win" — is solved by literal RC1 via `tauto`, confirming it was
correctly demoted in SX2 (within RC1's real reach, never a SET2 lever).

---

## 6. RC1 + `SET_ITE_SIMP` candidate

`candidate_results.json` — total 32, literal RC1 solved 11, candidate solved 16.

- **New wins over literal RC1 = 5**: `ite_empty_right`, `ite_right` (selected
  reproduction); `ite_empty`, `ite_empty_left`, `ite_left` (**fresh holdout, +3**).
- Gate fired on 10 theorems; precision: emitted_and_solved 5, emitted_and_failed 5,
  not_emitted 22.
- Emitted-and-failed (gate fired, `simp [Set.ite]` insufficient): `ite_inter`,
  `ite_inter_self`, `ite_eq_of_subset_left`, `ite_compl`, `ite_inter_of_inter_eq` —
  these need rw-bridges/per-branch proofs (search-depth gaps), correctly not claimed.
- **Off-gate emissions = 0. Regressions = 0** (additive — the candidate never ran a
  tactic on an RC1-solved theorem; RC1 behavior is untouched).

---

## 7. Minimal-sufficient attribution (NS23, vs literal RC1)

`minimal_relabel_results.json`:

| class | count |
|---|---|
| **TRUE_SET_ITE_SIMP_WIN** | **5** |
| RC1_ALREADY_SOLVED | 11 |
| NEEDS_DEEPER_SEQUENCE | 16 |
| BASELINE_DUPLICATE | 0 |
| PARSER_ARTIFACT | 0 |
| SOURCE_SPECIFIC | 0 |

All 5 wins: literal RC1 failed AND all four baselines failed (`baseline_solved_by =
None`) AND non-baseline `simp [Set.ite]` closed it. No duplicates, no parser
artifacts. The +3 fresh-holdout delta is fully confirmed.

---

## 8. Off-gate / preservation

`offgate_preservation_scan.json` — dry gate-only scan (pure name+goal predicate;
deterministic). 25 samples across nat_only / int_only / multiset / demo_v1 /
nat_defs_medium / set_positive_control + the negative-control and canonical sets.

- Gate emissions on non-Set surfaces: **0** → off-gate = 0.
- Positive Set controls fired: **2/2** (`Set.ite_right`, `Set.ite_empty_left`).
- The live candidate run independently confirms off-gate = 0 across all 32 theorems.

---

## 9. Determinism

`determinism_check.json` — candidate re-run on `known_wins` + `fresh_holdout`:
- run1 hash = run2 hash = `537e5b61ef2baf63`; per-theorem diffs = **0**;
  `deterministic = true`. (Candidate = deterministic literal-RC1 lookup + one fixed
  tactic, so identical signatures are expected.)

---

## 10. Decision

### `PROMOTE_TO_RC2_CANDIDATE` — narrow `SET_ITE_SIMP` only

| RC2 promotion requirement | status |
|---|---|
| positive delta over literal RC1 | ✅ **+5** (incl **+3 fresh holdout**) |
| zero regressions | ✅ 0 (additive) |
| zero off-gate emissions | ✅ 0 |
| minimal-sufficient attribution | ✅ 5/5 TRUE_SET_ITE_SIMP_WIN |
| deterministic reproduction | ✅ run1 == run2 |
| narrow gate only | ✅ one gate; six disabled |

All conditions pass against **literal RC1** (not a proxy). Per run discipline, the
recommendation is `PROMOTE_TO_RC2_CANDIDATE` — **not** a direct production merge.
Actual RC2 composition requires project-owner approval and canonical-floor
confirmation (§11). RC1 production configs remain untouched.

---

## 11. Next steps

**Positive path (recommended):**
1. Prepare an RC2 composition branch = **RC1 + `SET_ITE_SIMP` only** (additive
   namespace+ite gate, off-by-default emission logged; the 6 speculative gates stay
   disabled). Do not edit `rc1_production_wrapper.json` in place — compose a new RC2
   wrapper artifact.
2. Run canonical floors to confirm zero preservation regression: `demo_v1`,
   `nat_defs_medium`, `nat_defs_large_v5`.
3. Run the SF1 frontier benchmark to size total `Set.ite` headroom (every
   definitional-unfold `Set.ite_*` theorem RC1 fails).
4. The rw-bridge ite gaps (`ite_inter`, `ite_inter_self`, `ite_eq_of_subset_left`)
   remain search-depth gaps → motivate a future SX3 depth-limited sequence search,
   not a wrapper tactic.

**Negative path (not triggered here):** had the literal-RC1 delta been 0, the
candidate would be `TRAINING_DATA_ONLY`.

---

## 12. Protected-file confirmation

- `git diff --stat HEAD -- project/evolve/experiments/rc1/rc1_production_wrapper.json
  project/evolve/routing/ns24_router.json` → **empty** (untouched).
- NS9 genome/checkpoints, REL1 / RC1 release artifacts: untouched.
- `git status --short`: only new `??` `rc2_candidates/` + `scripts/rc2_*` + report
  (and pre-existing SF/SX artifacts + ` M README.md`).
- **No commit made.** All changes left in the working tree.
