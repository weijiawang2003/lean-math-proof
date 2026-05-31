# RC2 Candidate Wrapper — RC1 ⊕ SET_ITE_SIMP

`rc2_candidate_wrapper.json` is a **non-destructive composition**: a deep copy of
`project/evolve/experiments/rc1/rc1_production_wrapper.json` with exactly one added, schema-native component:

- `fallback_tactics += ["simp [Set.ite]"]`
- `theorem_name_tactic_gates += {"simp [Set.ite]": ['Set.ite']}`

## Why this preserves RC1
`theorem_name_tactic_gates` only filters **wrapper-added** entries (priority /
family / fallback / retrieved). Base-model (generative) output is never gated. The
literal string `simp [Set.ite]` is a substring of no other RC1 tactic, so the new
gate touches only the new fallback. On any theorem whose name does not start with
`Set.ite`, RC2's candidate set is byte-identical to RC1. On `Set.ite*` theorems it
adds one tactic. **Additive; RC1 behavior is never altered; regressions impossible
by construction.**

## Excluded (speculative SX2 gates, 0 true wins)
SET_EXT_SIMP, SET_SUBSET_ANTISYMM, SET_IFF_CONSTRUCTOR, SET_EXT_BYCASES, SET_RW_BRIDGE, SOURCE_SPECIFIC, broad_set_aesop.

## Status
`production_status = candidate` · `requires_owner_approval = true`. RC1/NS24 configs
are untouched. No commit.

## Run (full-wrapper eval)
```
python3 eval_rollout_all.py --theorem-set <name> --policy-type hybrid_evolved \
  --route-config project/evolve/routing/ns24_router.json \
  --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json \
  --top-k 8 --max-steps 8 --out-dir <out>
```
