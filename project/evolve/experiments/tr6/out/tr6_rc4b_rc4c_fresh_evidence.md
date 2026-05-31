# TR6 RC4B / RC4C fresh-holdout evidence

## RC4B — `Set.disjoint_left` bridge
- decision: **READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT**
- fresh true wins: 8 → ['Multiset.disjoint_add_left', 'Multiset.disjoint_cons_left', 'Multiset.singleton_disjoint', 'Multiset.zero_disjoint', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right']
- win namespaces: {'Multiset': 4, 'Set': 4}
- fire stats (top-5): {'fired_in_top5': 38, 'closed_as_credited': 8, 'fired_but_not_winning': 30}
- off-gate risk: low — single named-lemma rewrite gated to disjoint-shaped goals; fired in top-5 on 38 theorems, closed 8, fired-but-failed 30

## RC4C — `d2_simp_aesop`
- decision: **READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT**
- fresh true wins: 9 → ['Finset.biUnion_subset_iff_forall_subset', 'List.Forall.imp', 'Multiset.disjoint_add_left', 'Multiset.disjoint_add_right', 'Multiset.disjoint_cons_left', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right']
- win namespaces: {'Finset': 1, 'List': 1, 'Multiset': 3, 'Set': 4}
- fire stats (top-5): {'fired_in_top5': 135, 'closed_as_credited': 9, 'fired_but_not_winning': 126}
- overlap with RC4B: ['Multiset.disjoint_add_left', 'Multiset.disjoint_cons_left', 'Set.disjoint_iUnion_left', 'Set.disjoint_iUnion_right', 'Set.disjoint_sUnion_left', 'Set.disjoint_sUnion_right']
- source-specific risk: medium — credit is the simp[L] enabling step; SX4 PRODUCTION_SUBSUMED guard applied (RC2 confirmed-failure)
