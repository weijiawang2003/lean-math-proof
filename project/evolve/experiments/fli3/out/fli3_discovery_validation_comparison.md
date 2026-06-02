# FLI3 discovery → validation comparison

- FLI2 true rescues: 6 → FLI3 TRUE_FLI3_DELTA: 7
- **reproduced under literal RC2: 6/6 (rate 1.0)**
- not reproduced: []
- **family-holdout generalization wins: 1** ['List.bidirectionalRec_nil']
- families surviving: ['FINSET_CARD_BRIDGE', 'FINSET_MEM_DEF_UNFOLD', 'LIST_DEF_UNFOLD']
- control-duplicates: 0 | unknown-name gaps: 0

## Narrative

FLI2 discovered 6 robust rescues; under literal-RC2 additive validation 6 reproduce as TRUE_FLI3_DELTA (all). Family holdout produced 1 additional generalization wins, so failure-derived discovery DOES generalize beyond the exact discovered theorems. Failure-derived discovery CAN feed RC-style validation.
