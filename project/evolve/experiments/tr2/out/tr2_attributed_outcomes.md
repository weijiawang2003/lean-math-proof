# TR2 SX4 attribution

- cases: **34**  ·  **TRUE_DELTA: 0** []
- useful labels: **31**
- histogram: {'PROOF_SEARCH_DEPTH_GAP': 2, 'PRODUCTION_SUBSUMED': 3, 'MISSING_BRIDGE_LEMMA_CANDIDATE': 20, 'BASELINE_DUPLICATE': 6, 'NO_CHEAP_ACTION': 3}

| theorem | class | credit | useful | family | matches TR1? | reason |
|---|---|---|---|---|---|---|
| `Eq.subset` | **PROOF_SEARCH_DEPTH_GAP** | — | ✓ | depth_gap_bounded | False | bounded depth-2/3 battery + controls fail -> needs deeper se |
| `Multiset.disjoint_toFinset` | **PRODUCTION_SUBSUMED** | — | — | controls | None | RC2 solves it via multi-step search; no single cheap control |
| `Set.ite_eq_of_subset_left` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `coe_notMemRangeEquiv_symm` | **BASELINE_DUPLICATE** | — | ✓ | controls | True | RC2-solved; bare control also closes it -> routing/depth gap |
| `Set.pairwiseDisjoint_filter` | **PROOF_SEARCH_DEPTH_GAP** | — | ✓ | depth_gap_bounded | False | bounded depth-2/3 battery + controls fail -> needs deeper se |
| `Set.ite_eq_of_subset_right` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.antitoneOn_iff_antitone` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.monotoneOn_iff_monotone` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.pair_eq_pair_iff` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.subset_ite` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.union_empty_iff` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Function.Injective.nonempty_apply_iff` | **NO_CHEAP_ACTION** | — | ✓ | minimal_controls | True | minimal controls all fail on a confirmed RC2 failure -> veri |
| `Prop.compl_singleton` | **BASELINE_DUPLICATE** | — | ✓ | minimal_controls | False | confirmed RC2 failure but a bare control closes it (RC2 sear |
| `Multiset.toFinset_eq_singleton_iff` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | False | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.Nonempty.subset_pair_iff_eq` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.diff_singleton_subset_iff` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.ssubset_iff_insert` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.ssubset_iff_sdiff_singleton` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.ssubset_singleton_iff` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.subset_insert_iff` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.subset_pair_iff_eq` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.subset_singleton_iff_eq` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Multiset.toFinset_nsmul` | **BASELINE_DUPLICATE** | — | ✓ | controls | True | RC2-solved; bare control also closes it -> routing/depth gap |
| `Set.strictAntiOn_iff_strictAnti` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.strictMonoOn_iff_strictMono` | **MISSING_BRIDGE_LEMMA_CANDIDATE** | — | ✓ | retrieval | True | controls + exact? retrieval fail -> likely-existing Mathlib  |
| `Set.diff_union_inter` | **PRODUCTION_SUBSUMED** | — | — | controls | None | RC2 solves it via multi-step search; no single cheap control |
| `Set.insert_diff_eq_singleton` | **BASELINE_DUPLICATE** | — | ✓ | controls | True | RC2-solved; bare control also closes it -> routing/depth gap |
| `Set.insert_diff_of_mem` | **BASELINE_DUPLICATE** | — | ✓ | controls | True | RC2-solved; bare control also closes it -> routing/depth gap |
| `Set.ite_inter_of_inter_eq` | **NO_CHEAP_ACTION** | — | ✓ | minimal_controls | True | minimal controls all fail on a confirmed RC2 failure -> veri |
| `Set.eq_of_inclusion_surjective` | **NO_CHEAP_ACTION** | — | ✓ | minimal_controls | True | minimal controls all fail on a confirmed RC2 failure -> veri |
| `Set.inclusion_right` | **PRODUCTION_SUBSUMED** | — | — | controls | None | RC2 solves it via multi-step search; no single cheap control |
| `Set.pair_diff_left` | **BASELINE_DUPLICATE** | — | ✓ | controls | True | RC2-solved; bare control also closes it -> routing/depth gap |