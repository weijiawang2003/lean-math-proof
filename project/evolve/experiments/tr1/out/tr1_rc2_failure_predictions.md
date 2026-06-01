# TR1 router predictions on confirmed RC2 failures

- best model: `sgd`
- failures: **27**, held-out accuracy: **0.852** (23/27), abstained: 0

| theorem | true triage | predicted | score | abstain | next step |
|---|---|---|---|---|---|
| `Multiset.toFinset_eq_singleton_iff` | NO_CHEAP_ACTION | MISSING_BRIDGE_LEMMA_CANDIDATE | 0.967 | — | send to SF5 existing-lemma retrieval |
| `Set.antitoneOn_iff_antitone` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.diff_singleton_subset_iff` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.ite_eq_of_subset_left` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 0.55 | — | send to SF5 existing-lemma retrieval |
| `Set.ite_eq_of_subset_right` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 0.997 | — | send to SF5 existing-lemma retrieval |
| `Set.ite_inter_of_inter_eq` | NO_CHEAP_ACTION | NO_CHEAP_ACTION | 0.985 | — | send to lemma retrieval / deeper search |
| `Set.pair_eq_pair_iff` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.powerset_singleton` | NO_CHEAP_ACTION | NO_CHEAP_ACTION | 1.0 | — | send to lemma retrieval / deeper search |
| `Set.ssubset_singleton_iff` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.subset_insert_iff` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.subset_ite` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.subset_singleton_iff_eq` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.union_empty_iff` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Eq.subset` | NO_CHEAP_ACTION | PROOF_SEARCH_DEPTH_GAP | 0.591 | — | send to deeper bounded sequence search / widen aesop routing |
| `Function.Injective.nonempty_apply_iff` | NO_CHEAP_ACTION | NO_CHEAP_ACTION | 1.0 | — | send to lemma retrieval / deeper search |
| `Prop.compl_singleton` | PROOF_SEARCH_DEPTH_GAP | NO_CHEAP_ACTION | 1.0 | — | send to lemma retrieval / deeper search |
| `Set.Nonempty.subset_pair_iff_eq` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.eq_of_inclusion_surjective` | NO_CHEAP_ACTION | NO_CHEAP_ACTION | 1.0 | — | send to lemma retrieval / deeper search |
| `Set.monotoneOn_iff_monotone` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.pairwiseDisjoint_filter` | NO_CHEAP_ACTION | PROOF_SEARCH_DEPTH_GAP | 1.0 | — | send to deeper bounded sequence search / widen aesop routing |
| `Set.ssubset_iff_insert` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.ssubset_iff_sdiff_singleton` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.strictAntiOn_iff_strictAnti` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.strictMonoOn_iff_strictMono` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |
| `Set.subset_pair_iff_eq` | MISSING_BRIDGE_LEMMA_CANDIDATE | MISSING_BRIDGE_LEMMA_CANDIDATE | 1.0 | — | send to SF5 existing-lemma retrieval |