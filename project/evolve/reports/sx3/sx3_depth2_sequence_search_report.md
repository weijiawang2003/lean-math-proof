# SX3 Depth-2 Sequence Search — From SET_ITE_AESOP to General Sequence Mining

**Branch:** `sx3-depth2-sequence-search`  ·  **Date:** 2026-05-30  ·  **No commit made.**

---

## 1. Executive summary

- **SX3_SET_ITE_AESOP reproduced the +4 RC2-deferred wins: 4/4**, all via the depth-2
  sequence `simp [Set.ite] <;> aesop`, with every control (`simp`, `simp_all`, `aesop`,
  single-shot `simp [Set.ite]`) failing on each.
- **Fresh delta found: +1 genuine fresh true win — `Set.ite_inter_inter`** — solved by the
  same depth-2 sequence while all controls fail. `Set.ite_univ` looked like a candidate but is
  a **single-shot duplicate** (`simp [Set.ite]` alone closes it → belongs to RC2, excluded).
- **General depth-2 sequence search does NOT generalize.** Across 11 live general-Set cluster
  theorems, the `ext` / `iff-constructor` / `subset-antisymm` families produced **0** true
  depth-2 wins (only baseline duplicates).
- **Safety: 0 off-gate emissions** (negatives + smoke), **0 regressions vs RC2**, outcome-
  deterministic.
- **Decision: `RC3_CANDIDATE_CONFIRMED`** for `RC3 candidate = RC2 ⊕ SX3_SET_ITE_AESOP`
  (narrow `Set.ite` gate only, off-by-default, owner approval pending). The wrapper expresses
  the depth-2 sequence cleanly — `NEEDS_SEQUENCE_WRAPPER_SUPPORT` does not apply.
- **Recommendation:** keep the narrow gate; do not promote broad sequence search. Stand the
  candidate up for owner approval → RC3 literal-wrapper validation + full canonical floors.

---

## 2. Motivation

RC2 credited **+5 single-shot `simp [Set.ite]`** wins and **deferred +4** theorems
(`Set.ite_inter`, `Set.ite_inter_self`, `Set.ite_compl`, `Set.ite_inter_compl_self`) where
RC2-hardening forensics found that bare baselines and single-shot `simp [Set.ite]` all fail but
`simp [Set.ite] <;> aesop` succeeds. SX3 asks whether a **depth-2 sequence layer** can (a)
reproduce those +4 as a clean, attributable component, and (b) generalize to fresh Set.ite
surfaces — yielding an RC3 candidate distinct from RC2's single-shot delta.

**Attribution rule (never mixed):** RC2 official delta = single-shot `simp [Set.ite]`; SX3
candidate delta = depth-2 `simp [Set.ite] <;> aesop`.

---

## 3. Input audit

- Branch confirmed `sx3-depth2-sequence-search`; protected files present and byte-unchanged
  (RC1 / RC2-release wrappers, `ns24_router.json`) — see §13.
- RC2 wrapper: `project/evolve/experiments/rc2_release/rc2_production_wrapper.json`.
- Sources used (read-only): `rc2_hardening/out/rc2_delta_ledger.json` &
  `perturbation_forensics.json`; `sf2/out/set_cluster_deep_dive/selected_cases.json`;
  `sf1/out/real/{frontier_with_paths,classified_frontier,catalog}.jsonl`; RC2-candidate
  negative-control / canonical-smoke theorem sets.
- Audit written to `project/evolve/experiments/sx3/out/sx3_input_audit.json`.
- LeanDojo: traced cache `…mathlib4-29dcec…`; all Set.ite/dite theorems resolve to
  `Mathlib/Data/Set/Basic.lean`. `classical <;> aesop` parse-errors in `run_transition`
  (known limitation; bare `aesop` covers the same baseline).

---

## 4. Sequence registry

`project/evolve/experiments/sx3/sx3_sequence_registry.json` — 12 families, all off-by-default.

| family | sequence | source | RC3-eligible |
|---|---|---|---|
| **SX3_SET_ITE_AESOP** | `simp [Set.ite] <;> aesop` | rc2_deferred | **yes (candidate)** |
| SX3_SET_ITE_SIMPALL | `simp [Set.ite] <;> simp_all` | rc2_deferred | exploratory |
| SX3_SET_ITE_EXT | `ext x <;> simp [Set.ite]` | exploratory | exploratory |
| SX3_SET_ITE_EXT_AESOP | `ext x <;> simp [Set.ite] <;> aesop` | exploratory | exploratory |
| SX3_SET_EXT_AESOP | `ext x <;> aesop` | sf2_cluster | exploratory |
| SX3_SET_EXT_SIMPALL | `ext x <;> simp_all` | sf2_cluster | exploratory |
| SX3_SET_IFF_CONSTRUCTOR_AESOP | `constructor <;> intro h <;> aesop` | sf2_cluster | exploratory |
| SX3_SET_IFF_CONSTRUCTOR_SIMPALL | `constructor <;> intro h <;> simp_all` | sf2_cluster | exploratory |
| SX3_SET_SUBSET_ANTISYMM_AESOP | `apply Set.Subset.antisymm <;> intro x <;> aesop` | sf2_cluster | exploratory |
| SX3_SET_SUBSET_ANTISYMM_SIMPALL | `apply Set.Subset.antisymm <;> intro x <;> simp_all` | sf2_cluster | exploratory |
| SX3_SET_BYCASES_SIMPALL | `by_cases h : ?p <;> simp_all [h]` | diagnostic_only | no (needs ?p inference) |
| SX3_MULTISET_TOFINSET_AESOP | `simp [Multiset.mem_toFinset] <;> aesop` | diagnostic_only | no (future Multiset bridge) |

Only **SX3_SET_ITE_AESOP** is a plausible RC3 candidate; everything else is exploratory.

---

## 5. Deferred +4 reproduction

Runner: `scripts/sx3_run_depth2_sequences.py` (live LeanDojo, driver/worker with OS hard
timeout, per-tactic SIGALRM, Dojo-open guard, deterministic result hash).
Result: `out/sx3_deferred_results.json` (hash `c0144cd63fd5`, 4/4 live).

| theorem | `simp` | `simp_all` | `aesop` | `simp [Set.ite]` | `simp [Set.ite] <;> aesop` | class |
|---|---|---|---|---|---|---|
| Set.ite_inter | ✗ | ✗ | ✗ | ✗ | **✓** | new_depth2_win |
| Set.ite_inter_self | ✗ | ✗ | ✗ | ✗ | **✓** | new_depth2_win |
| Set.ite_compl | ✗ | ✗ | ✗ | ✗ | **✓** | new_depth2_win |
| Set.ite_inter_compl_self | ✗ | ✗ | ✗ | ✗ | **✓** | new_depth2_win |

All four: controls fail, depth-2 sequence solves → genuine depth-2 wins. **4/4 reproduced.**

---

## 6. Fresh Set.ite holdout

13 Set.ite/dite theorems from the Mathlib catalog, **excluding** RC2 credited-5 and deferred-4
(`out/sx3_fresh_holdout_results.json`, hash `ed6b9ef789a0`, 13/13 live).

| theorem | attribution | note |
|---|---|---|
| **Set.ite_inter_inter** | **TRUE_DEPTH2_SEQUENCE_WIN** | `simp [Set.ite] <;> aesop`; all controls fail → **fresh delta** |
| Set.ite_univ | SINGLE_STEP_DUPLICATE | single-shot `simp [Set.ite]` closes it → RC2, excluded |
| Set.mem_dite, mem_dite_empty_left/right, mem_dite_univ_left/right, mem_ite_empty_left/right (7) | BASELINE_DUPLICATE | bare `aesop` closes them |
| Set.ite_eq_of_subset_left/right, ite_inter_of_inter_eq, subset_ite (4) | NO_WIN | depth-2 sequence + controls all fail |

**Fresh true depth-2 wins: 1 — `Set.ite_inter_inter`.** The `ite_univ` single-shot duplicate
vindicates the strict attribution split (it is RC2's, not SX3's).

---

## 7. General Set cluster search

12 general (non-ite) Set failures (SF2 selected + frontier equality) × ext / iff-constructor /
subset-antisymm families (`out/sx3_set_cluster_results.json`, hash `071a3497ab3c`).

- 11/12 live (1 theorem, `Set.diff_singleton_subset_iff`, hit a lean_dojo Dojo-**open** hang and
  was OS-timed-out → `unknown`; this is an environment flake, see §8/§12).
- **0 true depth-2 wins.** 8 `no_sequence_win`, 3 `baseline_duplicate`.

**Conclusion:** the general ext/iff/subset families add nothing over the base policy on these
surfaces. Broad depth-2 sequence search does not generalize — consistent with the
namespace-saturation lesson (strong-base-policy Set surfaces are already covered by simp/aesop;
the depth-2 lever only pays on the narrow `simp[Set.ite]`-enabled `Set.ite` family).

---

## 8. Negative controls and canonical smoke

- **Negative controls** (`Nat.add_comm`, `Nat.mul_succ`, `Int.add_mul`,
  `Multiset.toFinset_eq_singleton_iff`, `Multiset.cons_inj_left`, `List.append_nil`):
  **0 Set-family emissions** — all gated out by `forbid_namespaces` / name token
  (`out/sx3_negative_control_results.json`, hash `754609f7e128`).
- **Canonical smoke** (`Nat.add_zero`, `Nat.zero_add`, `Nat.succ_le_succ`, `Bool.and_self`,
  `List.append_nil`): **0 Set-family emissions** (`out/sx3_canonical_smoke_results.json`,
  hash `aa47129f646c`).
- Most negative/smoke theorems carry synthetic `null` file paths so they don't resolve live
  (recorded `unknown`); this does **not** affect the off-gate guard, which is computed from the
  gate decision (emission) regardless of resolution. **Off-gate emissions = 0.**

---

## 9. Minimal attribution

`scripts/sx3_minimal_attribution.py` → `out/sx3_minimal_attribution.{json,md}`.

- **TRUE_DEPTH2_SEQUENCE_WIN = 5** — deferred `Set.ite_inter`, `Set.ite_inter_self`,
  `Set.ite_compl`, `Set.ite_inter_compl_self` + fresh `Set.ite_inter_inter`.
- **SINGLE_STEP_DUPLICATE = 1** (`Set.ite_univ` → RC2).  **BASELINE_DUPLICATE = 10.**
- **SOURCE_SPECIFIC = 0** — SX3 families are generic batteries (no rw bridge / theorem-specific
  lemma); structurally impossible, and confirmed.
- **off-gate = 0.**  **NEEDS_REVIEW = 11** = the unresolved null-path negatives/smoke (10) + the
  open-hung cluster theorem (1); none are suppressed Set.ite wins.
- Definition enforced: a true win = depth-2 sequence solved **and** every control
  (incl. single-shot `simp [Set.ite]` = RC2's credited mechanism) failed.

---

## 10. Family analysis

`scripts/sx3_analyze_sequence_families.py` → `out/sx3_family_analysis.{json,md}`.

| family | fresh | deferred | dup | off-gate | parse-err | score | recommendation |
|---|---|---|---|---|---|---|---|
| **SX3_SET_ITE_AESOP** | **1** | **4** | 0 | 0 | 0 | **6** | **RC3_CANDIDATE** |
| all other families | 0 | 0 | — | 0 | varies | ≤0 | KEEP_EXPERIMENTAL / REJECT_NO_DELTA |

**Best family: SX3_SET_ITE_AESOP.** No other family produced a true depth-2 win; none emitted
off-gate. The depth-2 lever is *specific* to the `simp[Set.ite]`-then-`aesop` shape.

---

## 11. RC3 candidate decision

### `RC3_CANDIDATE_CONFIRMED`

Candidate artifacts: `project/evolve/experiments/rc3_candidates/set_ite_aesop/`
(`rc3_candidate_wrapper.json`, `sx3_set_ite_aesop_gate.json`, `component_summary.json`,
`README.md`, `out/rc3_candidate_eval_results.json`, `out/rc3_candidate_comparison.json`).

Criteria (all met):

| criterion | result |
|---|---|
| positive fresh delta over RC2 | **+1** (`Set.ite_inter_inter`) |
| reproduces deferred +4 | **4/4** |
| 0 regressions vs RC2 | ✓ (0) |
| 0 off-gate | ✓ (0) |
| controls fail on every true win | ✓ |
| deterministic | ✓ (outcome-deterministic; see §12) |
| wrapper expresses sequence cleanly | ✓ — single-line `<;>` in `priority_templates["any"]` + `Set.ite` gate, placed after RC2's single-shot |

The candidate is byte-equivalent to RC2 on every theorem whose name lacks `Set.ite` (gate filters
only the wrapper-added entry; base output ungated), and only adds solves where the gated sequence
fires and RC2's single-shot/baselines fail.

---

## 12. Next steps & caveats

**Confirmed → next:**
1. Owner approval for `RC3 candidate = RC2 ⊕ SX3_SET_ITE_AESOP`.
2. RC3 literal-wrapper validation (run the candidate wrapper through the production eval harness
   on the Set.ite known/deferred/fresh/negative sets) to confirm the live wrapper reproduces the
   +5 depth-2 wins with 0 regressions, mirroring the RC2 literal-RC1 confirmation discipline.
3. Full canonical floors (demo_v1 / nat_defs_medium / nat_defs_large_v5) to confirm no floor
   regression before any production status.
4. Then (only then) prepare RC3 candidate production artifacts. **Do not call anything RC3
   production until validated and approved.**

**Caveats (honest):**
- **lean_dojo Dojo-open flake:** a few theorems (`Set.diff_singleton_subset_iff`,
  occasionally `Set.ite_inter_compl_self`) intermittently hang inside lean_dojo's session-open
  `time.sleep` retry loop, which SIGALRM does not always interrupt (lean_dojo manages its own
  signals); only the OS hard timeout advances them. This is **environment-level**, independent of
  SX3 logic — the affected theorems solve cleanly when they do open (and `Set.ite_inter_compl_self`
  is verified by the primary run + RC2-hardening direct probe). Concurrency makes it worse; runs
  were ultimately executed single-driver.
- **Determinism:** outcome-deterministic — every theorem that opened reproduced its classification
  and winning sequence byte-identically across runs (deferred re-run reproduced 3/3 opened
  theorems; the 4th flaked on open). Recorded in `out/sx3_determinism_recheck.json`.
- Runner fix during the run: concurrent runs originally shared `/tmp` worker paths (a race that
  leaked one cluster case into the fresh-holdout results); fixed with a per-run unique tmp prefix,
  and all reported results are from clean single-driver re-runs.

**If it had been experimental only:** keep as training data, mine more frontier clusters, improve
the open-hang robustness. Not the case here — the fresh delta is real.

---

## 13. Protected-file confirmation

`git diff --stat HEAD` for protected files is **empty** (unmodified):

- `project/evolve/experiments/rc1/rc1_production_wrapper.json` — untouched
- `project/evolve/experiments/rc2_release/rc2_production_wrapper.json` — untouched
  (source SHA `4573d9df6bcb` read-only; RC3 candidate is a separate functional copy)
- `project/evolve/routing/ns24_router.json` — untouched

All SX3 work is additive under `project/evolve/experiments/sx3/`,
`project/evolve/experiments/rc3_candidates/`, `project/evolve/reports/sx3/`, and `scripts/`.
**No commit made.** See §11 / `git status --short` in the final response for the full file list.
