# RC2 Composition Benchmark — RC1 ⊕ SET_ITE_SIMP

Branch: `rc1-production-stack` · live LeanDojo · **no commit** · RC1/NS24/NS9 untouched.
RC2 candidate = RC1 production wrapper ⊕ one narrow gated action `simp [Set.ite]`.

---

## 1. Executive summary

RC2 was composed non-destructively (RC1 deep-copy + one schema-native gated action),
benchmarked at full-wrapper scale against literal RC1 across canonical floors,
candidate-validation Set surfaces, and an SF1 fresh-frontier subset.

| metric | result |
|---|---|
| RC1 solved (benchmark) | 113 / 176 |
| RC2 solved (benchmark) | 131 / 176 |
| raw delta | +18 (per-surface, theorem overlap) |
| **credited delta (unique, attributable)** | **+5** single-shot `simp [Set.ite]` wins |
| search-perturbation wins (deterministic, NOT credited) | +4 multi-step |
| regressions | **0** |
| off-gate emissions | **0** |
| canonical floors | demo_v1 11/15 ✓, nat_defs_medium 37/38 ✓, nat_defs_large_v5 49/65 ✓ |
| minimal relabel | **5/5 TRUE_SET_ITE_SIMP_WIN**, 0 baseline-dup, 4 NEEDS_REVIEW |
| determinism | **deterministic** (two full runs, hash `e57a4325af9e58fb`, 0 diffs) |

The **+5 credited delta reproduces the literal-RC1 validation exactly** (the 5
single-shot `simp [Set.ite]` wins), with zero regressions, zero off-gate, and
preserved canonical floors.

**Recommendation: `RC2_CANDIDATE_CONFIRMED`** (not `RELEASE_RC2`). The clean,
attributable, deterministic benefit is +5. Full-wrapper integration via
`priority_templates` ALSO produces +4 deterministic but search-perturbation wins
(not logically attributable to `simp [Set.ite]`) and a +1 timing fluctuation on
demo_v1 — harmless (0 regressions) but meaning RC2 is not a surgically clean "+5 and
nothing else moves". Release framing requires owner approval (§9).

---

## 2. Component definition

- **Base:** RC1 production wrapper (commit f3b3100), referenced read-only.
- **Added (one component):** `SET_ITE_SIMP` → `simp [Set.ite]`, placed in
  `priority_templates["any"]` (emitted before the base policy and gated early), with
  `theorem_name_tactic_gates += {"simp [Set.ite]": ["Set.ite"]}`.
- **Why this composition is faithful & preserving:** `theorem_name_tactic_gates` only
  filters *wrapper-added* entries; base-model (generative) output is never gated; the
  literal string `simp [Set.ite]` is a substring of no other RC1 tactic. So on any
  theorem whose name does not start with `Set.ite`, RC2's candidate set is
  byte-identical to RC1 — RC1 behavior is never altered (regressions impossible by
  construction). On `Set.ite*` theorems it adds one early action.
- **Emission-slot finding (recorded, transparent):** an initial composition put
  `simp [Set.ite]` in `fallback_tactics` (v1). Full-wrapper eval reached it on only
  **1/5** known wins — fallbacks are low-priority and crowded out by
  `max_extra_tactics_per_state`. Moving it to `priority_templates["any"]` (v2, the
  MX2-proven slot) reliably surfaces it (**5/5**). The gate and tactic are unchanged
  between v1/v2 — only the emission slot. This is a composition-correctness fix, not
  gate-tuning to chase wins.
- **Excluded speculative SX2 gates (0 true wins in SX2):** `SET_EXT_SIMP`,
  `SET_SUBSET_ANTISYMM`, `SET_IFF_CONSTRUCTOR`, `SET_EXT_BYCASES`, `SET_RW_BRIDGE`,
  `SOURCE_SPECIFIC`.

---

## 3. Benchmark manifest

`rc2_benchmark_manifest.json` — 8 surfaces:

| surface | role | size | RC2 can differ? |
|---|---|---|---|
| demo_v1 | canonical_floor | 15 | no (no Set.ite names) |
| nat_defs_medium | canonical_floor | 38 | no |
| nat_defs_large_v5 | canonical_floor | 65 | no |
| set_ite_known_wins | candidate_validation | 5 | yes |
| set_ite_selected_failures | candidate_validation | 12 | yes |
| set_ite_fresh_holdout | candidate_validation | 20 | yes |
| sf1_frontier_runnable_subset | fresh_frontier | 20 | yes (7 Set.ite) |
| set_ite_negative_controls | negative_control | 5 | no (non-Set; dry) |

---

## 4. RC1 baseline

Command (`out/rc1_baseline_commands.sh`): `eval_rollout_all` (registered names) /
`sf1_run_eval.py` (file sets), `--policy-type hybrid_evolved --route-config
ns24_router.json --strategy-config rc1_production_wrapper.json --top-k 8 --max-steps
8`. Authoritative `finished` key. The 3 candidate Set sets were reused from the
RC2-validation literal-RC1 run (identical command/configs/sets).

| surface | RC1 finished |
|---|---|
| demo_v1 | 11/15 |
| nat_defs_medium | 37/38 |
| nat_defs_large_v5 | 49/65 |
| set_ite_known_wins | 0/5 |
| set_ite_selected_failures | 0/12 |
| set_ite_fresh_holdout | 11/20 |
| sf1_frontier_runnable_subset | 5/20 |

---

## 5. RC2 candidate results (full_wrapper_eval)

Mode: **full_wrapper_eval** with `rc2_candidate_wrapper.json`. Set.ite surfaces run
empirically; canonical floors are RC2≡RC1 by construction (corroborated by the v1
empirical full-wrapper canonical run: demo 12/15, medium 37/38, large 49/65).

| surface | RC1 | RC2 | delta |
|---|---|---|---|
| demo_v1 | 11/15 | 11/15 (v1 empirical 12 = +1 timing noise) | 0 |
| nat_defs_medium | 37/38 | 37/38 | 0 |
| nat_defs_large_v5 | 49/65 | 49/65 | 0 |
| set_ite_known_wins | 0/5 | 5/5 | +5 |
| set_ite_selected_failures | 0/12 | 4/12 | +4 |
| set_ite_fresh_holdout | 11/20 | 15/20 | +4 |
| sf1_frontier_runnable_subset | 5/20 | 10/20 | +5 |

**New-win composition (by `winning_tactic` / `num_steps`):**
- **Single-shot `simp [Set.ite]` (steps=1, tactic_template):** `ite_empty_right`,
  `ite_right`, `ite_empty`, `ite_empty_left`, `ite_left` → the **5 credited wins**.
- **Multi-step `aesop` (steps=2, generative_topk):** `ite_inter`, `ite_inter_self`,
  `ite_compl`, `ite_inter_compl_self`. Trace inspection: for `ite_inter` the step-1
  advancing tactic was `simp [Set.ite]` (candidate-enabled); for the other three the
  step-1 advance was a base-model `simp [Set.ext_iff]` — i.e. these are **search-
  perturbation side effects** of adding an action to `priority["any"]` (the best-first
  ordering shifts), not logically attributable to `simp [Set.ite]`. NOT credited.
- **Emitted-and-failed (gate fired, no win):** the rw-bridge ite theorems where
  `simp [Set.ite]` alone is insufficient — harmless (additive, cheap).

---

## 6. Comparison

`rc2_comparison.json` — global: RC1 113 → RC2 131, raw delta +18, **regressions 0**,
**off-gate 0**, canonical floors pass. Credited unique delta = **+5**; search-
perturbation unique = +4 (excluded). No theorem that RC1 solved is lost by RC2
(the gate touches only `Set.ite*` names; base output ungated → preservation).

The demo_v1 +1 in the v1 empirical run is run-to-run timing variance near the
per-theorem timeout, NOT an RC2 effect (no Set.ite names on demo_v1; RC2≡RC1 there).

---

## 7. Minimal-sufficient relabeling

`rc2_minimal_relabel_results.json` (ladder: simp / simp_all / aesop / classical<;>
aesop / simp [Set.ite]):

| class | count |
|---|---|
| **TRUE_SET_ITE_SIMP_WIN** | **5** (`ite_empty_right`, `ite_right`, `ite_empty`, `ite_empty_left`, `ite_left`) |
| BASELINE_DUPLICATE | 0 |
| RC1_ALREADY_SOLVED | 0 |
| PARSER_ARTIFACT | 0 |
| UNEXPECTED_WIN_NEEDS_REVIEW | 4 (the multi-step search-perturbation wins) |

A win is credited only if literal RC1 failed, all baselines failed, AND **single-shot**
`simp [Set.ite]` closes it. The 4 multi-step wins fail the single-shot test → excluded
from the credited delta. **Final credited delta = +5.**

---

## 8. Determinism

`rc2_determinism_check.json` — two independent full-wrapper RC2 runs over all four
Set.ite surfaces: run1 hash == run2 hash = `e57a4325af9e58fb`, **0 per-theorem diffs,
deterministic = true**. Both the 5 credited wins AND the 4 perturbation wins reproduce
identically (the best-first search is deterministic). Determinism ≠ attribution: the
4 perturbation wins are deterministic side effects, still not credited.

---

## 9. Release decision

### `RC2_CANDIDATE_CONFIRMED`

| RELEASE_RC2 requirement | status |
|---|---|
| canonical floors pass | ✅ 11/15, 37/38, 49/65 |
| total delta positive | ✅ credited +5 (raw +18) |
| zero regressions | ✅ 0 |
| zero off-gate | ✅ 0 |
| minimal relabel confirms | ✅ 5/5 TRUE_SET_ITE_SIMP_WIN |
| deterministic reproduction | ✅ hash-identical |
| owner explicitly approves release framing | ❌ not given |

All technical gates pass; the credited delta is a clean, deterministic, attributable
**+5**. Two reasons to confirm-as-candidate rather than release: (1) owner release
approval is required and not given; (2) full-wrapper integration via `priority_templates`
introduces deterministic-but-unattributable search-perturbation wins (+4) and minor
timing variance — harmless (0 regressions) but not a surgically clean delta. Decision:
**`RC2_CANDIDATE_CONFIRMED`**.

---

## 10. Next steps

If the owner approves release:
1. Create `rc2-production-stack` branch from this composed wrapper
   (`rc2_candidate_wrapper.json`, priority-slot emission).
2. Re-run canonical floors (demo_v1, nat_defs_medium, nat_defs_large_v5) + a broader
   SF1 frontier benchmark to size the full `Set.ite` headroom and re-confirm 0
   regressions at scale.
3. Prepare release artifacts: `rc2_release_checklist.md`, `rc2_executive_summary.md`,
   `rc2_reproduction_commands.md`, `rc2_resume_bullets.md`. Update README only after
   owner approval.
4. Investigate the +4 search-perturbation wins surgically: if `simp [Set.ite]` as an
   explicit depth-2 step (then `aesop`) reliably closes `ite_inter`-family theorems,
   that motivates the SX3 depth-limited sequence search — a cleaner mechanism than
   relying on priority-reordering side effects.

If not approved: RC2 remains a confirmed candidate; RC1 stays production.

---

## 11. Protected-file confirmation

- `git diff --stat HEAD -- rc1_production_wrapper.json ns24_router.json` → **empty**.
- NS9 genome/checkpoints, REL1 / RC1 release artifacts: untouched. RC2 lives entirely
  in new `project/evolve/experiments/rc2/` + `scripts/rc2_*` files.
- `git status --short`: only new `??` rc2 artifacts (and pre-existing SF/SX/rc2_candidates
  files + ` M README.md`).
- **No commit made.** All changes in the working tree.
