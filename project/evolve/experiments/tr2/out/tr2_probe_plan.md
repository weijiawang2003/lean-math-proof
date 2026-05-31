# TR2 probe plan

- theorems planned: **34**
- probe-family histogram: {'depth_gap_bounded': 2, 'controls': 8, 'retrieval': 20, 'minimal_controls': 4}
- routed to SF5 retrieval (no tactic spam): **20**

| theorem | rc2 | predicted | family | #probes | sf5? |
|---|---|---|---|---|---|
| `Eq.subset` | CONFIRMED_RC2_FAILURE | PROOF_SEARCH_DEPTH_GAP | depth_gap_bounded | 9 | — |
| `Multiset.disjoint_toFinset` | RC2_SOLVED | WX3_MULTISET_INDUCTION | controls | 4 | — |
| `Set.ite_eq_of_subset_left` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `coe_notMemRangeEquiv_symm` | RC2_SOLVED | BASELINE_DUPLICATE | controls | 4 | — |
| `Set.pairwiseDisjoint_filter` | CONFIRMED_RC2_FAILURE | PROOF_SEARCH_DEPTH_GAP | depth_gap_bounded | 9 | — |
| `Set.ite_eq_of_subset_right` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.antitoneOn_iff_antitone` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.monotoneOn_iff_monotone` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.pair_eq_pair_iff` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.subset_ite` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.union_empty_iff` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Function.Injective.nonempty_apply_iff` | CONFIRMED_RC2_FAILURE | NO_CHEAP_ACTION | minimal_controls | 3 | — |
| `Prop.compl_singleton` | CONFIRMED_RC2_FAILURE | NO_CHEAP_ACTION | minimal_controls | 3 | — |
| `Multiset.toFinset_eq_singleton_iff` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.Nonempty.subset_pair_iff_eq` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.diff_singleton_subset_iff` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.ssubset_iff_insert` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.ssubset_iff_sdiff_singleton` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.ssubset_singleton_iff` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.subset_insert_iff` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.subset_pair_iff_eq` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.subset_singleton_iff_eq` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Multiset.toFinset_nsmul` | RC2_SOLVED | NO_CHEAP_ACTION | controls | 4 | — |
| `Set.strictAntiOn_iff_strictAnti` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.strictMonoOn_iff_strictMono` | CONFIRMED_RC2_FAILURE | MISSING_BRIDGE_LEMMA_CANDIDATE | retrieval | 1 | Y |
| `Set.diff_union_inter` | RC2_SOLVED | BASELINE_DUPLICATE | controls | 4 | — |
| `Set.insert_diff_eq_singleton` | RC2_SOLVED | BASELINE_DUPLICATE | controls | 4 | — |
| `Set.insert_diff_of_mem` | RC2_SOLVED | BASELINE_DUPLICATE | controls | 4 | — |
| `Set.ite_inter_of_inter_eq` | CONFIRMED_RC2_FAILURE | NO_CHEAP_ACTION | minimal_controls | 3 | — |
| `Set.eq_of_inclusion_surjective` | CONFIRMED_RC2_FAILURE | NO_CHEAP_ACTION | minimal_controls | 3 | — |
| `Set.inclusion_right` | RC2_SOLVED | BASELINE_DUPLICATE | controls | 4 | — |
| `Set.pair_diff_left` | RC2_SOLVED | SET_ITE_SIMP | controls | 4 | — |