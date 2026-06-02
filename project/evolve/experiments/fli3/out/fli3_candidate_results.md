# FLI3 candidate eval

- items: 55 | gate fired: 24
- **candidate wins: 7** (robust 7) | rescue_replay 6/6 | family_holdout 1
- offgate emissions: 0 | regressions: 0 | unknown-name: 0

| set | theorem | gate | win | robust | winning tactic |
|---|---|---|---|---|---|
| family_holdout | `Finset.card_attach` | True | False | None | `` |
| family_holdout | `Finset.card_bij` | True | False | None | `` |
| family_holdout | `Finset.card_bij'` | True | False | None | `` |
| family_holdout | `Finset.card_bijective` | True | False | None | `` |
| family_holdout | `Finset.card_eq_of_bijective` | True | False | None | `` |
| family_holdout | `Finset.card_eq_one` | True | False | None | `` |
| family_holdout | `Finset.card_eq_succ` | True | False | None | `` |
| family_holdout | `Finset.exists_of_one_lt_card_pi` | True | False | None | `` |
| family_holdout | `Finset.filterMap_mono` | True | False | None | `` |
| family_holdout | `Finset.fin_map` | True | False | None | `` |
| family_holdout | `Finset.map_eq_of_subset` | True | False | None | `` |
| family_holdout | `Finset.map_ofDual_max` | True | False | None | `` |
| family_holdout | `Finset.map_ofDual_min` | True | False | None | `` |
| family_holdout | `Finset.map_toDual_max` | True | False | None | `` |
| family_holdout | `Finset.map_toDual_min` | True | False | None | `` |
| family_holdout | `Finset.preimage_subset` | True | False | None | `` |
| family_holdout | `List.bidirectionalRec_cons_append` | True | False | None | `` |
| family_holdout | `List.bidirectionalRec_nil` | True | True | True | `simp [List.bidirectionalRec]` |
| rescue_replay | `Finset.card_le_one_iff` | True | True | True | `simp [Finset.card_le_one] <;> aesop` |
| rescue_replay | `Finset.card_subtype` | True | True | True | `simp [Finset.subtype]` |
| rescue_replay | `Finset.mem_filterMap` | True | True | True | `simp [Finset.filterMap]` |
| rescue_replay | `Finset.mem_map` | True | True | True | `simp [Finset.map]` |
| rescue_replay | `Finset.mem_preimage` | True | True | True | `simp [Finset.preimage]` |
| rescue_replay | `List.bidirectionalRec_singleton` | True | True | True | `simp [List.bidirectionalRec]` |
