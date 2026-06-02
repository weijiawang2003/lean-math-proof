# FLI3 attribution

- items: 55 | classes: {'NO_DELTA': 17, 'GATE_NO_FIRE': 16, 'BASELINE_DUPLICATE': 15, 'TRUE_FLI3_DELTA': 7}
- **TRUE_FLI3_DELTA: 7** by family {'LIST_DEF_UNFOLD': 2, 'FINSET_CARD_BRIDGE': 1, 'FINSET_MEM_DEF_UNFOLD': 4} by set {'family_holdout': 1, 'rescue_replay': 6}
- rescue_replay reproduced: 6/6 | family_holdout wins: 1
- control-duplicates: 0 | unknown-name: 0 | flakes: 0

## TRUE_FLI3_DELTA

| theorem | family | lemma | tactic | set |
|---|---|---|---|---|
| `List.bidirectionalRec_nil` | LIST_DEF_UNFOLD | `List.bidirectionalRec` | `simp [List.bidirectionalRec]` | family_holdout |
| `Finset.card_le_one_iff` | FINSET_CARD_BRIDGE | `Finset.card_le_one` | `simp [Finset.card_le_one] <;> aesop` | rescue_replay |
| `Finset.card_subtype` | FINSET_MEM_DEF_UNFOLD | `Finset.subtype` | `simp [Finset.subtype]` | rescue_replay |
| `Finset.mem_filterMap` | FINSET_MEM_DEF_UNFOLD | `Finset.filterMap` | `simp [Finset.filterMap]` | rescue_replay |
| `Finset.mem_map` | FINSET_MEM_DEF_UNFOLD | `Finset.map` | `simp [Finset.map]` | rescue_replay |
| `Finset.mem_preimage` | FINSET_MEM_DEF_UNFOLD | `Finset.preimage` | `simp [Finset.preimage]` | rescue_replay |
| `List.bidirectionalRec_singleton` | LIST_DEF_UNFOLD | `List.bidirectionalRec` | `simp [List.bidirectionalRec]` | rescue_replay |
