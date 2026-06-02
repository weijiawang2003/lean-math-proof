# FLI3 rescue candidate summary

- **rescue candidates (robust TRUE rescues): 6** by family {'FINSET_MEM_DEF_UNFOLD': 4, 'FINSET_CARD_BRIDGE': 1, 'LIST_DEF_UNFOLD': 1}
- partial-progress candidates (separate): 29 by family {'FINSET_OTHER': 12, 'MULTISET_OTHER': 9, 'SET_OTHER': 6, 'LIST_OTHER': 2}

| id | theorem | family | lemma | tactic |
|---|---|---|---|---|
| FLI3-C01 | `Finset.card_le_one_iff` | FINSET_CARD_BRIDGE | `Finset.card_le_one` | `simp [Finset.card_le_one] <;> aesop` |
| FLI3-C02 | `Finset.card_subtype` | FINSET_MEM_DEF_UNFOLD | `Finset.subtype` | `simp [Finset.subtype]` |
| FLI3-C03 | `Finset.mem_filterMap` | FINSET_MEM_DEF_UNFOLD | `Finset.filterMap` | `simp [Finset.filterMap]` |
| FLI3-C04 | `Finset.mem_map` | FINSET_MEM_DEF_UNFOLD | `Finset.map` | `simp [Finset.map]` |
| FLI3-C05 | `Finset.mem_preimage` | FINSET_MEM_DEF_UNFOLD | `Finset.preimage` | `simp [Finset.preimage]` |
| FLI3-C06 | `List.bidirectionalRec_singleton` | LIST_DEF_UNFOLD | `List.bidirectionalRec` | `simp [List.bidirectionalRec]` |
