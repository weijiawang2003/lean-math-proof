# RC2 Candidate — `SET_ITE_SIMP` (literal-RC1 confirmation)

The ONLY RC2 candidate under validation: the narrow `SET_ITE_SIMP` gate
(`simp [Set.ite]`) on Set + ite/if goals. Carried forward from SX2, where it was
the only gate-worthy mined template (5 TRUE_SET2_WIN vs a 4-tactic baseline proxy).

This directory validates it against **literal RC1** (the unmodified
`rc1_production_wrapper.json` + `ns24_router.json` via `eval_rollout_all.py`,
`policy-type hybrid_evolved`, `top-k 8`, `max-steps 8`) — because the SX2 wins were
measured against an RC1-*proxy*, not the exact wrapper.

## Disabled / removed gates
`SET_EXT_SIMP`, `SET_EXT_BYCASES`, `SET_IFF_CONSTRUCTOR`, `SET_SUBSET_ANTISYMM`,
`SET_RW_BRIDGE`, `SOURCE_SPECIFIC` — all disabled (48 SX2 firings, 0 true wins).

## Layout
- `set_ite_simp_gate_policy.json` — single active gate (this candidate).
- `set_ite_simp_candidate_wrapper.json` — additive candidate descriptor.
- `theorem_sets/` — validation sets (known_wins, selected_failures, fresh_holdout,
  negative_controls, canonical_smoke) + `validation_manifest.json`.
- `out/` — literal RC1 results, candidate results, minimal relabel, off-gate scan,
  determinism check.

## Additive contract
`candidate_finished = literal_rc1_finished OR (gate_fired AND simp[Set.ite] solves)`.
RC1 behavior is never modified → regressions impossible by construction. The
candidate is off-by-default; nothing here promotes to production.

## Reproduce
See `scripts/rc2_*` and `out/literal_rc1_commands.sh`. No commit; protected RC1/NS24
configs untouched.
