# TR2 dataset delta

- TR1 examples: **57**  ·  combined: **57**
- **net-new: 0**  ·  reconfirmations: 27  ·  revision candidates: 4  ·  excluded: 3
- underrepresented improved: **False**
- new positive: []  ·  new negative: []

> Fresh frontier exhausted: every probed case already exists in TR1, so TR2 adds 0 net-new rows. Its value here is corroboration (27 reconfirmations) and 4 label-revision candidates flagged for human review — NOT dataset growth. To move TR1 off PILOT_ONLY_NEEDS_MORE_DATA, a genuinely fresh multi-namespace frontier must be sourced.

## Label distribution before → after

| label | before | after |
|---|---|---|
| BASELINE_DUPLICATE | 18 | 18 |
| MISSING_BRIDGE_LEMMA_CANDIDATE | 19 | 19 |
| NO_CHEAP_ACTION | 7 | 7 |
| PROOF_SEARCH_DEPTH_GAP | 1 | 1 |
| SET_ITE_SIMP | 6 | 6 |
| SX3_PRODUCTION_SUBSUMED | 5 | 5 |
| WX3_MULTISET_INDUCTION | 1 | 1 |

## Revision candidates (flagged, not applied)

| theorem | prior | new | reason |
|---|---|---|---|
| `Eq.subset` | NO_CHEAP_ACTION | PROOF_SEARCH_DEPTH_GAP | bounded depth-2/3 battery + controls fail -> needs deeper se |
| `Set.pairwiseDisjoint_filter` | NO_CHEAP_ACTION | PROOF_SEARCH_DEPTH_GAP | bounded depth-2/3 battery + controls fail -> needs deeper se |
| `Prop.compl_singleton` | PROOF_SEARCH_DEPTH_GAP | BASELINE_DUPLICATE | confirmed RC2 failure but a bare control closes it (RC2 sear |
| `Multiset.toFinset_eq_singleton_iff` | NO_CHEAP_ACTION | MISSING_BRIDGE_LEMMA_CANDIDATE | controls + exact? retrieval fail -> likely-existing Mathlib  |