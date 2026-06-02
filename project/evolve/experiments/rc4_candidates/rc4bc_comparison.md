# RC4B / RC4C joint candidate comparison

**Decision: `BOTH_READY_FOR_COMPOSITION`**

> Read-only comparison over the committed RC4B/RC4C validation record (commit `8f9d08e`). No live evaluation, no artifact mutation. Composition is a separate task.

## Side-by-side

| dimension | RC4B (disjoint_left bridge) | RC4C (d2 simp/aesop) |
|---|---|---|
| committed decision | `RC4B_CANDIDATE_CONFIRMED` | `RC4C_CONFIRMED_WITH_RC4B_OVERLAP` |
| TRUE_DELTA (total) | 16 | 15 (pure 7 + overlap-RC4B 8) |
| mechanism | `simp [<NS>.disjoint_left]  (optionally  <;> aesop) ; namespace-parametric over {Set, Multiset}` | `simp [L] <;> aesop over a small allowlist of retrieved lemmas L (per-lemma narrow gate)` |
| known wins | 11 | 12 |
| fresh-holdout wins | 8 (rate 0.727) | 9 (rate 0.75) |
| reproduction wins | 3 | 3 |
| namespace coverage (all) | {'Set': 7, 'Multiset': 9} | {'Finset': 1, 'List': 2, 'Multiset': 9, 'Set': 7} |
| off-gate emissions | 0 | 0 |
| regressions | 0 | 0 |
| offgate verdict | OFFGATE_CLEAN | OFFGATE_CLEAN |
| deterministic | ✅ `7574d704d3505a47` | ✅ `620a7e9d5dcdf044` |
| schema-smoke (known reproduced / no-regr) | 10/11, ✅ | 0/12, ✅ |
| added gated actions (wrapper simplicity) | 4 | 6 |

## Readiness checks

| check | RC4B | RC4C |
|---|---|---|
| has_true_delta | ✅ | ✅ |
| evidence_replays_probe | ✅ | ✅ |
| zero_regressions | ✅ | ✅ |
| zero_offgate | ✅ | ✅ |
| deterministic | ✅ | ✅ |
| wrapper_smoke_no_regression | ✅ | ✅ |
| **READY** | ✅ | ✅ |

### Caveats
- **RC4C:** production-search wrapper smoke reproduces 0 known wins under the deployed (fused `simp [L] <;> aesop`) form — deploy as the bare `simp [L]` enabling action (RC4B-style) so the search's own aesop closes; documented in RC4D.

## Overlap & residue

- RC4C overlaps RC4B on 8 wins (`disjoint_left` is literally an RC4B action); RC4C's net contribution beyond RC4B is 7 pure depth-2 wins via the residue actions ['simp [Multiset.disjoint_right] <;> aesop', 'simp [Set.subset_pair_iff_eq] <;> aesop', 'simp [Finset.biUnion_subset] <;> aesop', 'simp [List.forall_iff_forall_mem] <;> aesop'].
- RC4C residue actions beyond RC4B: `['simp [Multiset.disjoint_right] <;> aesop', 'simp [Set.subset_pair_iff_eq] <;> aesop', 'simp [Finset.biUnion_subset] <;> aesop', 'simp [List.forall_iff_forall_mem] <;> aesop']`

## Composition guidance (not performed here)

- If composed (separate task): order the gate as [RC4A, RC4B, RC4C_residue] so the namespace-parametric disjoint_left bridge (RC4B) is attributed before the depth-2 residue; de-duplicate the RC4B/RC4C overlap to avoid double-counting (8 Multiset/Set disjoint wins). This matches the already-committed RC4D composition.

## Decision

**`BOTH_READY_FOR_COMPOSITION`** — Both candidates clear every READY gate (>=1 TRUE_DELTA, probe evidence replay, 0 regressions, 0 off-gate, deterministic, wrapper-smoke no-regression). RC4C carries a deployment caveat (deploy bare `simp [L]`, not the fused combinator) and overlaps RC4B; its additive value for an RC4 composition is the residue actions only.

_Both candidates remain off-by-default; this comparison promotes nothing and composes nothing. The committed RC4D composition already realizes the guidance above._
