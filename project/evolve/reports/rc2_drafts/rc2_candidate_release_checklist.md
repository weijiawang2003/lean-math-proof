# RC2 Candidate — Release Checklist (DRAFT, NOT executed)

> This is a draft. Do NOT update the README production recommendation or create a
> release commit until the owner explicitly approves RC2 release framing.

## Technical gates (all PASS at candidate validation)
- [x] Reproduction stable (3 independent full-wrapper runs; credited +5 identical)
- [x] Credited delta +5 over **literal** RC1 (single-shot `simp [Set.ite]`)
- [x] Minimal-sufficient relabel: 5/5 TRUE_SET_ITE_SIMP_WIN, 0 baseline-duplicate
- [x] 0 regressions (RC2 ≡ RC1 on every non-`Set.ite` theorem, by construction)
- [x] 0 off-gate emissions (gate name-prefixed to `Set.ite`)
- [x] Canonical floors preserved: demo_v1 11/15, nat_defs_medium 37/38, large_v5 49/65
- [x] Deterministic (hash-stable across runs)
- [x] No speculative gates present in the candidate wrapper
- [x] +4 perturbation wins formally classified (SX3 depth-2 sequence candidates, deferred)

## Owner-gated steps (NOT done)
- [ ] Owner approves RC2 release framing
- [ ] (optional) Broader SF1 `Set.ite` frontier sweep to size total headroom
- [ ] Create `rc2-production-stack` branch from `rc2_candidate_wrapper.json`
- [ ] Freeze the RC2 wrapper artifact + re-run full preservation at scale
- [ ] Update README production recommendation (RC1 → RC2)
- [ ] Prepare release commit + artifacts

## Artifacts to freeze on approval
- `project/evolve/experiments/rc2/rc2_candidate_wrapper.json` (deployable, priority slot)
- `project/evolve/experiments/rc2/rc2_component_summary.json`
- This report set + `project/evolve/reports/rc2_hardening_attribution_report.md`

## Explicit non-goals for RC2
- The +4 depth-2 sequence wins (`simp [Set.ite] <;> aesop`) are NOT part of RC2's
  credited delta — they belong to a separate SX3 sequence-search candidate requiring
  its own literal-RC1 + minimal-relabel validation.
