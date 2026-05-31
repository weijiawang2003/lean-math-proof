# RC2 Candidate — Executive Summary (DRAFT, pending owner approval)

**RC2 = RC1 ⊕ narrow `SET_ITE_SIMP`** (`simp [Set.ite]`), gated to `Set.ite*` theorem
names. Additive, off-by-default, composed non-destructively (RC1 deep-copy + one
schema-native gated action in `priority_templates["any"]`). Speculative SX2 gates
excluded. RC1 production configs untouched.

## Clean credited result
**+5 true single-shot `SET_ITE_SIMP` wins over literal RC1**, confirmed across three
independent full-wrapper runs (hash-stable) and a minimal-sufficient relabel:

- `Set.ite_empty_right`, `Set.ite_right`, `Set.ite_empty`, `Set.ite_empty_left`,
  `Set.ite_left` — each failed by literal RC1 AND all four baselines
  (simp/simp_all/aesop/classical<;>aesop), closed by single-shot `simp [Set.ite]`.

## Preservation
- Canonical floors preserved: demo_v1 11/15, nat_defs_medium 37/38,
  nat_defs_large_v5 49/65 (RC2 ≡ RC1 on every non-`Set.ite` theorem, by construction).
- **0 regressions, 0 off-gate emissions.** Deterministic (hash-stable across runs).

## Additional observation (excluded from official delta)
The deployable full-wrapper placement also closes 4 further theorems
(`Set.ite_inter`, `Set.ite_inter_self`, `Set.ite_compl`, `Set.ite_inter_compl_self`)
by using `simp [Set.ite]` as an enabling *step* followed by `aesop`. Live forensic
probes prove these are genuine **depth-2 sequence wins** (`simp [Set.ite] <;> aesop`
closes them; bare `aesop` and single-shot `simp [Set.ite]` do not). They are
**deferred to a separate SX3 depth-2 sequence candidate**, NOT counted in the RC2
single-shot credited delta.

## Release status
**Candidate confirmed; release-ready with caveat — pending owner approval.** The
clean attribution-bearing delta is +5. For perturbation-free attribution, an additive
single-shot evaluator (Variant D) reproduces exactly +5; the deployable schema-native
wrapper (Variant A, `priority_templates`) reproduces +5 and additionally delivers the
4 depth-2 sequence wins. Optional next step before release: a broader SF1 `Set.ite`
frontier sweep to size total headroom.
