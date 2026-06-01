# RC4B — disjoint_left bridge evidence

- known wins (deduped): **11**
- by namespace: {'Multiset': 4, 'Set': 7}
- by bridge lemma: {'Multiset.disjoint_left': 4, 'Set.disjoint_left': 7}
- TR5 reproduction evidence: **3** ['Set.disjoint_iff_forall_ne', 'Set.disjoint_right', 'Set.disjoint_singleton_left']
- TR6 fresh evidence: **8** ['Multiset.disjoint_add_left', 'Multiset.disjoint_cons_left', 'Multiset.singleton_disjoint', 'Multiset.zero_disjoint', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right']
- needs_review (excluded): 0

| theorem | ns | bridge | winning tactic | source | fresh | class |
|---|---|---|---|---|---|---|
| `Multiset.disjoint_add_left` | Multiset | `Multiset.disjoint_left` | `simp [Multiset.disjoint_left] <;> aesop` | TR6 | True | FRESH_TRUE_DELTA |
| `Multiset.disjoint_cons_left` | Multiset | `Multiset.disjoint_left` | `simp [Multiset.disjoint_left] <;> aesop` | TR6 | True | FRESH_TRUE_DELTA |
| `Multiset.singleton_disjoint` | Multiset | `Multiset.disjoint_left` | `simp [Multiset.disjoint_left]` | TR6 | True | FRESH_TRUE_DELTA |
| `Multiset.zero_disjoint` | Multiset | `Multiset.disjoint_left` | `simp [Multiset.disjoint_left]` | TR6 | True | FRESH_TRUE_DELTA |
| `Set.disjoint_iUnion_left` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR6 | True | FRESH_TRUE_DELTA |
| `Set.disjoint_iUnion_right` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR6 | True | FRESH_TRUE_DELTA |
| `Set.disjoint_iff_forall_ne` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR3+TR5 | False | TRUE_DELTA |
| `Set.disjoint_right` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR3+TR5 | False | TRUE_DELTA |
| `Set.disjoint_sUnion_left` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR6 | True | FRESH_TRUE_DELTA |
| `Set.disjoint_sUnion_right` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left] <;> aesop` | TR6 | True | FRESH_TRUE_DELTA |
| `Set.disjoint_singleton_left` | Set | `Set.disjoint_left` | `simp [Set.disjoint_left]` | TR3+TR5 | False | TRUE_DELTA |

## Mechanism

simp [<NS>.disjoint_left]  (optionally  <;> aesop) ; namespace-parametric over {Set, Multiset}

All wins share one mechanism — a single named-lemma rewrite with `<NS>.disjoint_left` that turns `Disjoint a b` into a membership goal closable by simp/aesop. The only axis of variation is the namespace (Set vs Multiset), which the gate keys on; so a single narrow candidate with two namespace variants covers them. NOT CANDIDATE_TOO_HETEROGENEOUS.
