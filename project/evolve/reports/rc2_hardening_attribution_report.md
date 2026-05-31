# RC2 Hardening + Attribution Cleanup Report

Branch: `rc1-production-stack` · live LeanDojo · **no commit** · RC1/NS24/NS9 untouched.
Candidate: RC2 = RC1 ⊕ narrow `SET_ITE_SIMP` (`simp [Set.ite]`).

---

## 1. Executive summary

The RC2 candidate **reproduced cleanly** and its attribution is now fully resolved.

| item | result |
|---|---|
| reproduction (3rd independent full-wrapper run) | identical: known 5/5, selected 4/12, holdout 15/20, frontier 10/20 |
| credited delta | **+5** single-shot `simp [Set.ite]` wins (stable, minimal-relabel TRUE 5/5) |
| +4 "perturbation" wins | **forensically reclassified as SX3 depth-2 sequence candidates** (not artifacts), deferred |
| regressions | 0 | off-gate | 0 | canonical floors | pass (11/15, 37/38, 49/65) |
| determinism | hash-stable across runs |
| best attribution-clean integration | Variant D (additive single-shot) — +5, zero perturbation |
| deployable wrapper | Variant A (`priority_templates`) — +5, schema-native (also yields the 4 sequence wins) |

**Decision: `RC2_RELEASE_READY_WITH_CAVEAT`** (owner approval still required for actual
release/README change). All technical gates pass; the +4 are formally excluded from the
credited delta and explained as a clean SX3 line.

---

## 2. Reproduction

Fresh full-wrapper RC2 run on the four Set.ite surfaces (RC1 baseline reused; canonical
floors are RC2≡RC1 by construction). Per-theorem signature identical to the two prior
runs. `reproduction_comparison.json`: credited unique **+5**
(`Set.ite_empty`, `Set.ite_empty_left`, `Set.ite_empty_right`, `Set.ite_left`,
`Set.ite_right`), 0 regressions, 0 off-gate, floors pass. Reproduction is **stable** —
release framing is permitted to proceed.

---

## 3. Perturbation forensics (the +4)

`perturbation_forensics.json` — live direct probes (authoritative) on the +4:

| theorem | single-shot `simp [Set.ite]` | `simp [Set.ite] <;> aesop` | bare `aesop` | `simp_all` | verdict |
|---|---|---|---|---|---|
| Set.ite_inter | ✗ | ✓ | ✗ | ✗ | SX3 sequence candidate |
| Set.ite_inter_self | ✗ | ✓ | ✗ | ✗ | SX3 sequence candidate |
| Set.ite_compl | ✗ | ✓ | ✗ | ✗ | SX3 sequence candidate |
| Set.ite_inter_compl_self | ✗ | ✓ | ✗ | ✗ | SX3 sequence candidate |

**All four are genuine depth-2 sequence wins:** `simp [Set.ite] <;> aesop` closes each
one, while bare `aesop`/`simp_all` AND single-shot `simp [Set.ite]` all fail. So
`simp [Set.ite]` is a true *enabling step* (it transforms the goal so `aesop` can
finish) — not a search-order coincidence. In the RC2 full wrapper these closed via
`simp [Set.ite]` at step 1 then `aesop` at step 2 (the same depth-2 program realized
across two search steps). **SX3 implication:** a depth-limited sequence search seeded by
`SET_ITE_SIMP` (then `aesop`) is the right mechanism for this family — cleaner than
relying on priority-reorder side effects, and it should be validated on its own.

These are **not** counted in the RC2 single-shot credited delta.

---

## 4. Surgical integration variants

`variant_comparison.json` (probe pass over the 5 credited + 4 sequence theorems;
canonical floors unaffected by placement, by construction):

| variant | credited recovered | extra wins | regr | off-gate | perturbation | deployable | schema-native |
|---|---|---|---|---|---|---|---|
| A `priority_templates["any"]` | 5/5 | +4 (depth-2 sequence, via search) | 0 | 0 | yes (reorders base) | **yes** | **yes** |
| D additive single-shot | 5/5 | 0 | 0 | 0 | **none** | no (eval mode) | no |
| E `simp [Set.ite] <;> aesop` | 5/5 | +4 (all 9) | 0 | 0 | none | no (SX3) | no |

- **Attribution-clean reference:** Variant D recovers exactly +5 with zero search
  perturbation — the cleanest basis for the *official* credited delta.
- **Deployable artifact:** Variant A (`rc2_candidate_wrapper.json`) is the only
  schema-native `eval_rollout_all` wrapper; it recovers +5 and, as a bonus, closes the
  4 depth-2 sequence theorems through search (deterministic, 0 regression).
- **Variant E** is the SX3 depth-2 sequence candidate (solves all 9) — NOT RC2; needs
  its own literal-RC1 + minimal-relabel validation.
- Variants B (late-priority) and C (fallback cap-fix) were not run: B offers no
  attribution benefit over A (still full-wrapper perturbation); C needs per-state-cap
  schema support that does not exist. Documented, not silently dropped.

Chosen: **official delta from D's semantics (+5, perturbation-free); deploy via A**
(schema-native, +5 plus harmless deterministic sequence bonus).

---

## 5. Credited-delta ledger

`rc2_delta_ledger.json` — category histogram `{credited_SET_ITE_SIMP: 5,
SX3_sequence_candidate: 4}`, excluded 0.

| decision | theorems |
|---|---|
| **credit** (credited_SET_ITE_SIMP) | `Set.ite_empty_right`, `Set.ite_right`, `Set.ite_empty`, `Set.ite_empty_left`, `Set.ite_left` |
| **defer** (SX3_sequence_candidate) | `Set.ite_inter`, `Set.ite_inter_self`, `Set.ite_compl`, `Set.ite_inter_compl_self` |
| exclude | — (none) |

**Official RC2 credited delta = +5.** No win is a baseline-duplicate, parser artifact,
or timeout-variance case.

---

## 6. Preservation / off-gate

`preservation_hardening.json` — `hardening_ok = true`:
- speculative gates present: **NONE** (SET_EXT_SIMP / SUBSET_ANTISYMM / IFF_CONSTRUCTOR
  / EXT_BYCASES / RW_BRIDGE / SOURCE_SPECIFIC all absent).
- off-gate emissions on Nat/Int/Multiset/Bool/List surfaces: **0**; positive Set.ite
  controls fire 2/2.
- canonical floors pass: demo_v1 11/15, nat_defs_medium 37/38, nat_defs_large_v5 49/65.
- regressions across all benchmark surfaces: **0**.
- gate emits only on `Set.ite*` names (name-prefix gate).

---

## 7. Draft release docs

Created under `project/evolve/reports/rc2_drafts/` (DRAFTS — README untouched, no
release framing applied):
`rc2_candidate_executive_summary.md`, `rc2_candidate_reproduction_commands.md`,
`rc2_candidate_release_checklist.md`, `rc2_candidate_resume_bullets.md`.

---

## 8. Decision

### `RC2_RELEASE_READY_WITH_CAVEAT`

| condition | status |
|---|---|
| reproduction stable | ✅ (3 hash-stable runs) |
| credited +5 stable | ✅ |
| 0 regressions | ✅ |
| 0 off-gate | ✅ |
| floors pass | ✅ 11/15, 37/38, 49/65 |
| perturbation wins formally excluded/explained | ✅ deferred as SX3 depth-2 sequence candidates |

**Caveat (explicit):** the official credited delta is **+5 single-shot** `simp [Set.ite]`
wins. The deployable `priority_templates` wrapper additionally closes 4 depth-2
sequence theorems through search — deterministic and regression-free, but attributable
to a `simp [Set.ite] <;> aesop` *sequence*, not the single tactic; these belong to a
separate SX3 line. Actual release (branch, wrapper freeze, README update, release
commit) remains **gated on explicit owner approval**.

---

## 9. Next steps

If the owner approves release:
1. Create `rc2-production-stack` from `rc2_candidate_wrapper.json` (priority slot).
2. Freeze the wrapper; re-run full preservation at scale + a broader SF1 `Set.ite`
   frontier sweep.
3. Update README (RC1 → RC2) and prepare the release commit + checklist execution.

In parallel / if not approved:
4. Stand up **SX3**: a depth-limited sequence search seeded by `SET_ITE_SIMP` then
   `aesop`, validated on its own (literal-RC1 + minimal relabel) to formally claim the
   4 deferred sequence wins.

---

## 10. Protected-file confirmation

- `git diff --stat HEAD -- rc1_production_wrapper.json ns24_router.json` → **empty**.
- NS9 genome/checkpoints, REL1 / RC1 release artifacts untouched. README untouched
  (no production-recommendation change).
- `git status --short`: only new `??` `rc2_hardening/` + `rc2_drafts/` + `scripts/rc2_*`
  (+ pre-existing experiment files + ` M README.md`).
- **No commit made.**
