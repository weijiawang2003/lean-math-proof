# RC3 Literal-Wrapper Validation — RC2 ⊕ SX3_SET_ITE_AESOP

Validates the **RC3 candidate** through the **production eval harness** (`eval_rollout_all.py`
via `scripts/sf1_run_eval.py`), not only the custom SX3 runner. Confirms the credited SX3
depth-2 delta is real over a **literal RC2** baseline measured on identical theorem sets.

## Candidate definition

    RC3 candidate = RC2 release wrapper  ⊕  SX3_SET_ITE_AESOP

Exactly **one** added component beyond RC2:

- `priority_templates["any"]`: add `simp [Set.ite] <;> aesop` **immediately after** `simp [Set.ite]`
  (RC2's credited single-shot is always tried first; the depth-2 sequence only contributes
  when the single-shot fails).
- `theorem_name_tactic_gates`: add `"simp [Set.ite] <;> aesop": ["Set.ite"]`.

Every other key is **byte-identical** to the RC2 release wrapper. The gate filters only
wrapper-added entries by theorem-name substring, so the candidate is byte-equivalent to RC2
on every theorem whose name lacks `Set.ite`.

### Gate (narrow)

- name signal: `Set.ite`
- forbid namespaces: `Nat`, `Int`, `Multiset`, `List`
- max one SX3 emission per theorem
- **no** broad Set sequence search; **no** ext/iff/subset exploratory families;
  **no** source-inspired theorem-specific `rw` bridges.

## Protected / untouched

- `project/evolve/experiments/rc1/rc1_production_wrapper.json`
- `project/evolve/experiments/rc2_release/rc2_production_wrapper.json`
- `project/evolve/routing/ns24_router.json`
- NS9 genome/checkpoints, REL1/RC1/RC2 release reports

The RC3 candidate is a **separate** off-by-default config, not wired into `ns24_router.json`.

## Files

| File | Purpose |
|---|---|
| `rc3_candidate_wrapper.json` | the candidate strategy config (byte-identical copy of the source candidate) |
| `rc3_component_summary.json` | exact structural delta vs RC2 + provenance |
| `rc3_validation_manifest.json` | Part-1 freeze manifest (candidate + untouched + pointers) |
| `validation_manifest.json` | operational manifest: theorem sets, sizes, leakage, known wins/no-wins |
| `theorem_sets/` | the six validation theorem sets |
| `out/` | live run results, attribution, preservation/off-gate, determinism, comparison |

## Theorem sets (`theorem_sets/`)

| set | size | runnable | role |
|---|---|---|---|
| `sx3_known_deferred` | 4 | 4 | +4 RC2-deferred reproduction target (not fresh delta) |
| `sx3_fresh_win` | 1 | 1 | `Set.ite_inter_inter` (subset of holdout) |
| `sx3_set_ite_holdout` | 13 | 13 | all fresh Set.ite/dite holdout (wins + no-wins) |
| `sx3_negative_controls` | 6 | 1 | non-Set off-gate guard |
| `sx3_canonical_smoke` | 5 | 0 | non-Set canonical sample (floors run as registered sets) |
| `sx3_set_cluster_cases` | 12 | 12 | general Set cluster where broad families failed |

## Reproduce

See `out/literal_rc2_commands.sh` and `out/literal_rc3_commands.sh` for exact commands.
Pipeline scripts: `scripts/rc3_run_literal_validation.py`, `scripts/rc3_minimal_relabel_set_ite_aesop.py`,
`scripts/rc3_preservation_offgate.py`, `scripts/rc3_determinism_flake_audit.py`,
`scripts/rc3_compare_validation.py`. Report: `project/evolve/reports/rc3/rc3_set_ite_aesop_validation_report.md`.
