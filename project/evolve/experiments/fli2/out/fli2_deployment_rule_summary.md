# FLI2 deployment rule summary

- mined rules: 25 (candidate: 1)
- supporting rescues total: 11

| rule | ns | family | actions | rescues | partials | FP | risk | status |
|---|---|---|---|---|---|---|---|---|
| FINSET_MAP_BRIDGE | Finset | Finset.map_* | SIMPLE_SIMP,SIMP_AESOP | 4 | 0 | 45 | medium | needs_more_data |
| FINSET_IMAGE_BRIDGE | Finset | Finset.image_* | SIMPLE_SIMP,SIMP_AESOP | 2 | 1 | 6 | medium | needs_more_data |
| FINSET_SUBTYPE_BRIDGE | Finset | Finset.subtype_* | SIMPLE_SIMP,SIMP_AESOP | 2 | 0 | 0 | low | candidate |
| LIST_BIDIRECTIONALREC_BRIDGE | List | List.bidirectionalrec_* | SIMPLE_SIMP,SIMP_AESOP | 2 | 0 | 6 | medium | needs_more_data |
| FINSET_CARD_BRIDGE | Finset | Finset.card_* | SIMP_AESOP | 1 | 6 | 145 | medium | needs_more_data |
| FINSET_MEM_BRIDGE | Finset | Finset.mem_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 15 | 21 | medium | needs_more_data |
| LIST_SINGLETON_BRIDGE | List | List.singleton_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 6 | 6 | medium | needs_more_data |
| MULTISET_MAP_BRIDGE | Multiset | Multiset.map_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 5 | 85 | medium | needs_more_data |
| SET_IMAGE_BRIDGE | Set | Set.image_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 5 | 43 | medium | needs_more_data |
| SET_SUBSET_BRIDGE | Set | Set.subset_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 5 | 52 | medium | needs_more_data |
| MULTISET_MEM_BRIDGE | Multiset | Multiset.mem_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 4 | 13 | medium | needs_more_data |
| FINSET_INSERT_BRIDGE | Finset | Finset.insert_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 2 | 4 | medium | needs_more_data |
| MULTISET_BIND_BRIDGE | Multiset | Multiset.bind_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 2 | 8 | medium | needs_more_data |
| MULTISET_FILTER_BRIDGE | Multiset | Multiset.filter_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 2 | 49 | medium | needs_more_data |
| FINSET_BIUNION_BRIDGE | Finset | Finset.biunion_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 15 | medium | needs_more_data |
| FINSET_CLOSER_BRIDGE | Finset | closer | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 32 | medium | needs_more_data |
| FINSET_COE_BRIDGE | Finset | Finset.coe_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 1 | medium | needs_more_data |
| FINSET_SUBSET_BRIDGE | Finset | Finset.subset_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 41 | medium | needs_more_data |
| LIST_SUBSET_BRIDGE | List | List.subset_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 3 | medium | needs_more_data |
| MULTISET_ATTACH_BRIDGE | Multiset | Multiset.attach_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 1 | medium | needs_more_data |
| MULTISET_CARD_BRIDGE | Multiset | Multiset.card_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 5 | medium | needs_more_data |
| MULTISET_COUNT_BRIDGE | Multiset | Multiset.count_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 3 | medium | needs_more_data |
| SET_IINTER_BRIDGE | Set | Set.iinter_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 8 | medium | needs_more_data |
| SET_IUNION_BRIDGE | Set | Set.iunion_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 28 | medium | needs_more_data |
| SET_LEFTINVON_BRIDGE | Set | Set.leftinvon_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 1 | medium | needs_more_data |
