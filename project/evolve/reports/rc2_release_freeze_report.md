# RC2 Release-Freeze Report — RC1 ⊕ SET_ITE_SIMP

Branch: `rc1-production-stack` · owner-approved release prep · **no commit** (per
instruction, commit only after this report is shown). RC1 / NS24 / NS9 / REL1 untouched.

---

## 1. Executive summary

RC2 is **release-frozen**. RC2 = RC1 ⊕ one narrowly gated action `simp [Set.ite]`,
composed non-destructively in `priority_templates["any"]` and gated to `Set.ite*`
theorem names. The frozen production wrapper was re-verified live and behaves
identically to the validated candidate.

- **Production config:** `project/evolve/experiments/rc2_release/rc2_production_wrapper.json`
- **Official credited delta:** **+5** single-shot `SET_ITE_SIMP` wins over literal RC1.
- **Safety:** 0 regressions, 0 off-gate emissions, canonical floors preserved,
  deterministic.
- **Caveat:** 4 additional depth-2 `Set.ite` sequence wins are deferred to SX3 and
  excluded from the official delta (headline is +5, not the raw surface-summed +18).

---

## 2. Artifact list

**New RC2 release files** (`project/evolve/experiments/rc2_release/`):
- `rc2_production_wrapper.json` — frozen production wrapper (RC1 ⊕ SET_ITE_SIMP + release metadata).
- `rc2_component_summary.json`, `rc2_reproduction_config.json`, `README.md`
- `final_verification.json` / `.md`

**New release reports** (`project/evolve/reports/`):
- `rc2_release_checklist.md`, `rc2_executive_summary.md`, `rc2_reproduction_commands.md`,
  `rc2_resume_bullets.md`, `rc2_attribution_notes.md`, `rc2_release_freeze_report.md` (this file).

**README:** added a "Recommended production stack: RC2" section; RC1 preserved as the
previous baseline (history not erased).

**Preserved unchanged:** `rc1_production_wrapper.json`, `ns24_router.json`, NS9
genome/checkpoints, REL1 reports.

---

## 3. Final verification (`final_verification.json/md`) — overall_pass: **true**

| check | result |
|---|---|
| JSON validity (wrapper, summary, repro config) | all valid |
| protected diff (RC1 wrapper + NS24 router) | empty (untouched) |
| canonical floor demo_v1 | 12/15 (≥11) — live, frozen wrapper |
| canonical floor nat_defs_medium | 37/38 (≥37) — RC2≡RC1 by construction |
| canonical floor nat_defs_large_v5 | 49/65 (≥49) — RC2≡RC1 by construction |
| SET_ITE known wins | 5/5 (+5) — live, frozen wrapper |
| determinism smoke (known_wins ×2) | hash-stable (`bbbd688b72d00c06`), 0 diffs |

**Reused vs rerun:** demo_v1 + the 5 SET_ITE wins were rerun LIVE with the frozen
production wrapper; nat_defs_medium / nat_defs_large_v5 are RC2≡RC1 by construction
(the gate denies the added action on every non-`Set.ite` name; base-model output is
never gated) and match the RC1 baseline (37/38, 49/65) — corroborated by the full
benchmark RC2 run earlier this line of work. demo_v1's 12 vs RC1's 11 is run-to-run
timing variance near the per-theorem timeout (no `Set.ite` names on demo_v1), not an
RC2 effect.

---

## 4. Attribution

- **Credited +5** (single-shot `simp [Set.ite]`, literal-RC1-confirmed, minimal-relabel
  5/5 TRUE): `Set.ite_empty_right`, `Set.ite_right`, `Set.ite_empty`,
  `Set.ite_empty_left`, `Set.ite_left`.
- **Deferred to SX3 (+4, NOT credited):** `Set.ite_inter`, `Set.ite_inter_self`,
  `Set.ite_compl`, `Set.ite_inter_compl_self` — depth-2 sequence wins
  (`simp [Set.ite] <;> aesop`; bare `aesop`/`simp_all` and single-shot `simp [Set.ite]`
  all fail).
- **No raw +18 headline.** The raw surface-summed +18 double-counts shared theorems and
  folds in the deferred depth-2 wins; the de-duplicated, single-shot-attributed figure
  is +5. See `rc2_attribution_notes.md`.

---

## 5. Reproduction command

```
python3 eval_rollout_all.py --theorem-set <set> \
  --policy-type hybrid_evolved \
  --route-config project/evolve/routing/ns24_router.json \
  --strategy-config project/evolve/experiments/rc2_release/rc2_production_wrapper.json \
  --top-k 8 --max-steps 8 --out-dir <run-dir>
```

---

## 6. Commit readiness

- **Files changed:** new `rc2_release/` artifacts + 6 release reports + README section
  (RC2 recommendation). All other prior-task artifacts remain in the working tree.
- **Protected files unchanged:** `rc1_production_wrapper.json`, `ns24_router.json`,
  NS9 genome/checkpoints, REL1 reports — all empty diff.
- **No commit made.** Working tree is commit-ready pending owner go-ahead.
- **Suggested commit message:** `Prepare RC2 release artifacts`
