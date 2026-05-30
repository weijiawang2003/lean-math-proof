# RC2 Release — RC1 ⊕ SET_ITE_SIMP (frozen)

Release-frozen production candidate. **RC2 = RC1 ⊕ one narrow gated action**
`simp [Set.ite]`. Composed non-destructively; RC1 / NS24 / NS9 / REL1 untouched.

## Composition
`RC2 = NS9 base wrapper ⊕ WX3 Multiset induction oracle ⊕ MX2 narrow
Set.Finite/toFinset aesop fallback ⊕ SET_ITE_SIMP narrow Set.ite gate`

The only delta versus `rc1_production_wrapper.json`:
- `priority_templates["any"]` prepends `simp [Set.ite]`
- `theorem_name_tactic_gates` adds `{"simp [Set.ite]": ["Set.ite"]}`

`theorem_name_tactic_gates` filters only *wrapper-added* entries; base-model output
is never gated; `simp [Set.ite]` is a substring of no other RC1 tactic. So RC2 is
**byte-identical to RC1 on every non-`Set.ite` theorem** (regressions impossible by
construction) and adds one early action on `Set.ite*` theorems.

## Files
- `rc2_production_wrapper.json` — frozen production wrapper (this is the `--strategy-config`).
- `rc2_component_summary.json` — composition + credited delta + safety.
- `rc2_reproduction_config.json` — exact eval params + floors.
- `final_verification.json` / `.md` — final verification results.

## Official credited delta
**+5** single-shot `SET_ITE_SIMP` wins over literal RC1:
`Set.ite_empty_right`, `Set.ite_right`, `Set.ite_empty`, `Set.ite_empty_left`,
`Set.ite_left`. Minimal-relabel: 5/5 TRUE, 0 baseline-duplicate.

## Safety
0 regressions · 0 off-gate emissions · canonical floors preserved
(demo_v1 11/15, nat_defs_medium 37/38, nat_defs_large_v5 49/65) · deterministic.

## Caveat (deferred to SX3, NOT in official delta)
The deployable wrapper also deterministically closes 4 depth-2 sequence theorems
(`Set.ite_inter`, `Set.ite_inter_self`, `Set.ite_compl`, `Set.ite_inter_compl_self`)
via `simp [Set.ite]` then `aesop`. Forensics prove `simp [Set.ite] <;> aesop` closes
them while bare `aesop`/`simp_all` and single-shot `simp [Set.ite]` fail → genuine
**SX3 depth-2 sequence candidates**, excluded from the RC2 single-shot credited delta.

## Run
```
python3 eval_rollout_all.py --theorem-set <set> \
  --policy-type hybrid_evolved \
  --route-config project/evolve/routing/ns24_router.json \
  --strategy-config project/evolve/experiments/rc2_release/rc2_production_wrapper.json \
  --top-k 8 --max-steps 8 --out-dir <run-dir>
```
RC1 remains preserved as the previous baseline at
`project/evolve/experiments/rc1/rc1_production_wrapper.json`.
