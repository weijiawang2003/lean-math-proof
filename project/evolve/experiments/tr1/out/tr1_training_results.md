# TR1 training results

- examples: **57**, features: 1600
- CV: LeaveOneOut OOF (tiny corpus; honest per-sample held-out)
- **best model: `sgd` (macro-F1 0.628)**
- beats rule baseline: **True**

## Model comparison

| model | accuracy | macro_F1 | top3 | grouped(LONO) acc |
|---|---|---|---|---|
| rule_baseline | 0.632 | 0.436 | — | — |
| logistic | 0.842 | 0.579 | 0.965 | 0.404 |
| sgd | 0.877 | 0.628 | 0.93 | 0.386 |
| random_forest | 0.789 | 0.462 | 0.965 | 0.105 |

## Label support

- `BASELINE_DUPLICATE`: 18
- `MISSING_BRIDGE_LEMMA_CANDIDATE`: 19
- `NO_CHEAP_ACTION`: 7
- `PROOF_SEARCH_DEPTH_GAP`: 1
- `SET_ITE_SIMP`: 6
- `SX3_PRODUCTION_SUBSUMED`: 5
- `WX3_MULTISET_INDUCTION`: 1

## Best model per-label (sgd)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `BASELINE_DUPLICATE` | 0.944 | 0.944 | 0.944 | 18 |
| `MISSING_BRIDGE_LEMMA_CANDIDATE` | 0.95 | 1.0 | 0.974 | 19 |
| `NO_CHEAP_ACTION` | 0.8 | 0.571 | 0.667 | 7 |
| `PROOF_SEARCH_DEPTH_GAP` | 0.0 | 0.0 | 0.0 | 1 |
| `SET_ITE_SIMP` | 0.857 | 1.0 | 0.923 | 6 |
| `SX3_PRODUCTION_SUBSUMED` | 1.0 | 0.8 | 0.889 | 5 |
| `WX3_MULTISET_INDUCTION` | 0.0 | 0.0 | 0.0 | 1 |