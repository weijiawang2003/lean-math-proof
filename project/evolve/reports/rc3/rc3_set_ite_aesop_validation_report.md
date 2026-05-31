# RC3 Literal-Wrapper Validation — RC2 ⊕ SX3_SET_ITE_AESOP

**Branch:** `sx3-depth2-sequence-search`  ·  **Date:** 2026-05-30  ·  **No commit made.**

---

## 1. Executive summary

| field | value |
|---|---|
| RC3 candidate | RC2 release ⊕ `SX3_SET_ITE_AESOP` (one added component: `simp [Set.ite] <;> aesop`, narrow `Set.ite` gate) |
| validation surface | 30 theorems (deferred 4 · fresh 1 · holdout 13 · cluster 12 · neg-control 1, deduped by name) |
| literal RC2 solved | **17 / 30** |
| literal RC3 solved | **17 / 30** |
| raw delta | **+0** |
| **credited SX3 delta over literal RC2** | **0** |
| fresh true wins | **0** |
| regressions vs RC2 | **0** |
| off-gate emissions | **0** |
| canonical floors | demo_v1 12/15 · medium 37/38 · large 49/65 — **all pass** |
| determinism | **deterministic** (run1 hash `5757bbb27215` = run2, 0 open flakes, 0 proof diffs) |
| **DECISION** | **`REJECT_NO_LITERAL_DELTA`** |

**The RC3 candidate is safe but adds nothing.** Through the **production eval harness**,
literal RC2 *already* solves all four "deferred" theorems **and** the "fresh"
`Set.ite_inter_inter` — every one of them via a natural **2-step best-first search path**:
`simp [Set.ite]` at step 1 (advances the goal) then `aesop` at step 2 (closes it). That two-step
chain *is* the SX3 depth-2 sequence, discovered by ordinary search. Adding the explicit grouped
tactic `simp [Set.ite] <;> aesop` produces **zero new wins**, because the wins are not new.

The custom SX3 runner over-credited the sequence because it tested controls **single-shot from the
initial state only** (depth-1): bare `aesop` on the *original* goal fails, so the runner concluded
the grouped sequence was a unique enabling step. It never tested `aesop` on the
`simp [Set.ite]`-*advanced* state — which is exactly what the production search does at step 2.
This is the same subsumption that SX1 found for general depth-2 sequences, now confirmed for the
`Set.ite` case against a literal RC2 baseline.

The candidate remains **clean**: 0 regressions, 0 off-gate emissions, canonical floors preserved,
fully deterministic. It is simply **subsumed** — there is no literal delta to credit, so it should
**not** become an RC3 release candidate. Keep RC2 release as the production stack.

---

## 2. Background

- **RC2 official delta = +5 single-shot.** RC2 release (frozen) added `SET_ITE_SIMP`: the
  single-shot tactic `simp [Set.ite]` gated to `Set.ite`-named theorems, crediting +5 wins
  (`Set.ite_empty`, `…_empty_left/right`, `…_left`, `…_right`).
- **SX3 depth-2 search** (custom runner) claimed +5 *additional* "true depth-2 wins" for the
  sequence `simp [Set.ite] <;> aesop`: the 4 RC2-hardening "deferred" theorems plus a fresh
  `Set.ite_inter_inter`, with all single-shot controls reportedly failing.
- **Why validate over literal RC2.** Goal A of this task: confirm the claim through the *production*
  harness, not the proxy runner. A custom runner that only probes depth-1 controls cannot see what a
  max-steps-8 / top-k-8 best-first search finds at depth 2. The literal run resolves the question.

---

## 3. Candidate definition

**Sequence added:** `simp [Set.ite] <;> aesop`

**Exact structural delta vs RC2 (everything else byte-identical):**

1. `priority_templates["any"]`: insert `simp [Set.ite] <;> aesop` **immediately after**
   `simp [Set.ite]` (RC2's single-shot is always tried first).
2. `theorem_name_tactic_gates`: add `"simp [Set.ite] <;> aesop": ["Set.ite"]`.

**Gate (narrow):** name signal `Set.ite`; forbid `Nat`/`Int`/`Multiset`/`List`; max 1 SX3 emission
per theorem; **no** broad Set sequence search; **no** ext/iff/subset exploratory families; **no**
source-inspired theorem-specific `rw` bridges. The gate filters only wrapper-added entries, so the
candidate is byte-equivalent to RC2 on every theorem whose name lacks `Set.ite`.

**Excluded families** (SX3 found 0 true depth-2 wins for them): `SET_ITE_SIMPALL`, `SET_ITE_EXT`,
`SET_ITE_EXT_AESOP`, `SET_EXT_AESOP/SIMPALL`, `SET_IFF_CONSTRUCTOR_*`, `SET_SUBSET_ANTISYMM_*`,
`SET_BYCASES_SIMPALL`, `MULTISET_TOFINSET_AESOP`.

---

## 4. Validation theorem sets

`project/evolve/experiments/rc3_validation/theorem_sets/` (deduped by `full_name` for the live run):

| set | size | runnable | role |
|---|---|---|---|
| `sx3_known_deferred` | 4 | 4 | +4 RC2-deferred reproduction target |
| `sx3_fresh_win` | 1 | 1 | `Set.ite_inter_inter` (⊂ holdout) |
| `sx3_set_ite_holdout` | 13 | 13 | all fresh Set.ite/dite holdout (wins + no-wins) |
| `sx3_negative_controls` | 6 | 1 | non-Set off-gate guard (5 are gate-structural, null file_path) |
| `sx3_canonical_smoke` | 5 | 0 | non-Set sample (floors run as registered sets instead) |
| `sx3_set_cluster_cases` | 12 | 12 | general Set cluster where broad families failed |

Leakage handled: `Set.ite_univ` (single-shot dup), `Set.ite_inter_of_inter_eq` (rw-bridge source).
`Set.ite_inter_inter` deduped across `sx3_fresh_win`/`sx3_set_ite_holdout`. Full manifest:
`project/evolve/experiments/rc3_validation/validation_manifest.json`.

---

## 5. Literal RC2 baseline

Command: `out/literal_rc2_commands.sh` (policy `hybrid_evolved`, route `ns24_router.json`,
strategy `rc2_release/rc2_production_wrapper.json`, top-k 8, max-steps 8).
Results: `out/literal_rc2_results.json` — **17/30 finished, 30/30 available (0 open flakes)**.

| role | solved | unsolved |
|---|---|---|
| deferred_known (4) | **4** | 0 |
| fresh_win (1) | **1** | 0 |
| fresh_holdout (12 ex-fresh) | 8 | 4 (`ite_eq_of_subset_left/right`, `ite_inter_of_inter_eq`, `subset_ite`) |
| set_cluster_failure (12) | 4 | 8 |
| negative_control (1) | 0 | 1 (`Multiset.toFinset_eq_singleton_iff`) |

**Key:** every deferred + fresh theorem is solved by RC2 in **2 steps**, winning tactic `aesop`:

```
Set.ite_inter            steps=2  tactics=['simp [Set.ite]', 'aesop']   win=aesop  origin=generative_topk
Set.ite_inter_self       steps=2  tactics=['simp [Set.ite]', 'aesop']   win=aesop
Set.ite_compl            steps=2  tactics=['simp [Set.ite]', 'aesop']   win=aesop
Set.ite_inter_compl_self steps=2  tactics=['simp [Set.ite]', 'aesop']   win=aesop
Set.ite_inter_inter      steps=2  tactics=['simp [Set.ite]', 'aesop']   win=aesop
```

Trace (`Set.ite_inter`): step 1 `simp [Set.ite]` → `TacticState` (1 goal → 1 goal, advanced);
step 2 from the advanced state `aesop` → `ProofFinished`. The RC2 traces contain **0** instances of
the grouped sequence tactic — RC2 doesn't have it; it reaches the same result by chaining two search
steps.

---

## 6. Literal RC3 candidate

Command: `out/literal_rc3_commands.sh` (identical except `--strategy-config rc3_candidate_wrapper.json`).
Results: `out/literal_rc3_results.json` — **17/30 finished, 30/30 available**.

- **New wins over literal RC2: 0.**
- **Regressions vs RC2: 0.**
- The grouped sequence `simp [Set.ite] <;> aesop` is added to the priority battery and is **gate-eligible**
  on the deferred/fresh theorems, but it is **never the winning tactic**: the search still closes via
  the decomposed `simp [Set.ite]` (step 1) → `aesop` (step 2) path first. RC3 ≡ RC2 on the entire surface.
- Emitted-and-failed: on the 4 unsolved `Set.ite`-named holdout theorems
  (`ite_eq_of_subset_left/right`, `ite_inter_of_inter_eq`, `subset_ite`) the gate fires and the
  sequence is tried but does not close them — consistent with SX3's own `NO_WIN` attribution.
- No parse errors; no Dojo open flakes (30/30 available in both runs).

---

## 7. Minimal-sufficient attribution

`scripts/rc3_minimal_relabel_set_ite_aesop.py` → `out/rc3_minimal_relabel_results.{json,md}`.

For every RC3 new win over **literal** RC2, the script opens a fresh Dojo and runs the control battery
(`simp`, `simp_all`, `aesop`, `classical <;> aesop`, `simp [Set.ite]`, `simp [Set.ite] <;> aesop`),
then classifies (`TRUE_SX3_SET_ITE_AESOP_WIN` / `SINGLE_STEP_DUPLICATE` / `BASELINE_DUPLICATE` /
`SOURCE_SPECIFIC` / `PARSER_ARTIFACT` / `OPEN_FLAKE` / `NEEDS_REVIEW`).

- **RC3 new wins over literal RC2: 0** → **0 live probes run.**
- **TRUE_SX3_SET_ITE_AESOP_WIN: 0.**
- The 5 previously-claimed wins (4 deferred + 1 fresh) are **`RC2_ALREADY_SOLVED`** and excluded
  up front — they are wins of the *literal RC2 best-first search*, not of the added sequence.
- No parser/open-flake artifacts (no probes needed).

**Credited SX3 delta over literal RC2 = 0.**

> Root cause of the prior over-credit: the SX3 runner's `aesop` control was applied single-shot to
> the *initial* goal (fails), never to the `simp [Set.ite]`-advanced state (succeeds). The production
> harness applies `simp [Set.ite]` then `aesop` as two ordinary search steps, so the "enabling step"
> the grouped sequence supposedly provided was already provided by the search.

---

## 8. Preservation / off-gate

`scripts/rc3_preservation_offgate.py` → `out/rc3_preservation_offgate.{json,md}`.

| floor | RC3 solved | total | floor min | RC2 doc | pass | regression | off-gate |
|---|---|---|---|---|---|---|---|
| demo_v1 | 12 | 15 | 11 | 12 | ✅ | 0 | 0 |
| nat_defs_medium | 37 | 38 | 37 | 37 | ✅ | 0 | 0 |
| nat_defs_large_v5 | 49 | 65 | 49 | 49 | ✅ | 0 | 0 |

- **All floors pass; 0 regressions vs the documented RC2 release counts.**
- **Off-gate emissions: 0.** Off-gate = the grouped sequence emitted on a theorem whose name lacks
  `Set.ite`. Across all three floors (incl. pure-Nat medium/large = live non-Set evidence) and the
  negative controls: **0**. demo_v1 emits the sequence only on its one `Set.ite`-named theorem
  (`Set.ite_univ`, on-gate) and still closes it via the single-shot, count unchanged.
- **Negative control** `Multiset.toFinset_eq_singleton_iff`: available, not solved, **0 off-gate**.
  The five null-`file_path` controls (`Nat.add_comm`, `Nat.mul_succ`, `Int.add_mul`,
  `Multiset.cons_inj_left`, `List.append_nil`) are not live-openable; off-gate for them is asserted
  structurally (gate is the name substring `Set.ite`) and corroborated by the pure-Nat floors.
- No timeout variance; 0 open flakes on the floors.

---

## 9. Determinism and flake audit

`scripts/rc3_determinism_flake_audit.py` → `out/rc3_determinism_flake_audit.json`. RC3 run **twice**
(sequentially) on deferred + fresh + holdout (17 unique theorems).

| field | run 1 | run 2 |
|---|---|---|
| result hash | `5757bbb27215` | `5757bbb27215` |

- **hash match: True**; per-theorem `finished` + `winning_tactic` identical.
- **open flakes: 0** (17/17 opened in both runs); **proof-result diffs: 0**.
- **Classification: `deterministic`.** The LeanDojo open-time flakes noted in the SX3 report did **not**
  recur on this surface in this audit; had they, every theorem that opened in both runs was still
  identical, which would have yielded `deterministic_except_environment_open_flake`. Reported honestly:
  no flakes were hidden — none occurred.

---

## 10. Decision

### `REJECT_NO_LITERAL_DELTA`

Criteria for `RC3_RELEASE_CANDIDATE_CONFIRMED`:

| criterion | met? |
|---|---|
| positive credited delta over literal RC2 | ❌ (0) |
| at least one fresh true win | ❌ (0) |
| 0 regressions | ✅ |
| 0 off-gate | ✅ |
| canonical floors pass | ✅ |
| minimal attribution confirms | ❌ (0 true wins) |
| deterministic or isolated env-open flake only | ✅ (deterministic) |

The candidate is **safe** but provides **no incremental wins** over a literal RC2 run, because RC2's
production best-first search already discovers the equivalent 2-step `simp [Set.ite]` → `aesop` path.
The added grouped sequence is **subsumed**. This is not a regression or an off-gate failure or a
nondeterminism failure — it is the absence of a literal delta.

---

## 11. Next steps

**RC3 is not release-candidate-ready** — there is nothing to release: the wins already belong to RC2.

- **Keep RC2 release unchanged** as the production deterministic stack
  (`rc2_release/rc2_production_wrapper.json`). Do **not** stand up an `rc3_release`.
- **Keep SX3 experimental / off-by-default.** The candidate wrapper stays where it is, documented as
  *subsumed by the production search* (this report).
- **Update the SX3 / RC3-candidate status** from `RC3_CANDIDATE_CONFIRMED` to
  `SUBSUMED_BY_PRODUCTION_SEARCH` (no literal delta). The "+4 deferred + 1 fresh" should be recorded
  as **wins of the literal RC2 full-wrapper best-first search**, not of an added sequence action.
- **Methodology fix for future sequence mining:** measure depth-2 candidates against a **literal
  production run** (multi-step search), not a depth-1 single-shot control battery. A depth-1 control
  battery structurally cannot see depth-2 search closes and will over-credit grouped sequences.
- **Frontier mining** remains the open lever where the base policy is genuinely weak (per memory:
  Multiset quotient surfaces), not where the search already chains short tactics (Set.ite).

---

## 12. Protected-file confirmation

See `out/protected_files_check.txt` for the live `git diff --stat` and `git status --short`.
RC1 wrapper, RC2 release wrapper, and NS24 router show **no diff** (untouched). All new files are
under `project/evolve/experiments/rc3_validation/`, `project/evolve/reports/rc3/`, and
`scripts/rc3_*`. **No commit was made.**
