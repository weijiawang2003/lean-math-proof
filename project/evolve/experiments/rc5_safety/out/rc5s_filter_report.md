# RC5S existing-plan filter

- original programs: 1792 → filtered (allowed): **1269** (removed 523)
- classification: {'POLICY_ALLOWED': 1269, 'REMOVED_STALL_RISK': 366, 'REMOVED_OFF_POLICY': 73, 'REMOVED_NAMESPACE_DISABLED': 84}
- off-policy removed: 73 | stall-risk removed: 366 | namespace-disabled: 84
- **all 3 RC5H true-hybrid winners survive: True**

## Winner survival

| theorem | winning tactic | class | survives |
|---|---|---|---|
| `Finset.biUnion_subset_iff_forall_subset` | `simp [Finset.biUnion_subset] <;> aesop` | POLICY_ALLOWED | True |
| `Finset.image_subset_iff` | `simp [Finset.subset_iff]` | POLICY_ALLOWED | True |
| `Multiset.add_bind` | `simp [Multiset.bind]` | POLICY_ALLOWED | True |
