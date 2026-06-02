# FLI3 Validation Atlas

## 1. Overview
Validated 55 items; literal RC2 {'solved': 16, 'failed': 39}; **TRUE_FLI3_DELTA 7** (rescue_replay 6/6, family_holdout 1).

## 2. Why FLI3 follows FLI2
FLI2 discovered 6 robust at-position rescues; FLI3 tests whether they survive RC-style literal-RC2 additive validation and whether the families generalize.

## 3. Candidate families
FINSET_CARD_BRIDGE, FINSET_MEM_DEF_UNFOLD, LIST_DEF_UNFOLD.

## 4. Validation sets
55 items across rescue_replay/family_holdout/offgate_negative/canonical_floor/regression_guard.

## 5. Literal RC2 baseline
{'solved': 16, 'failed': 39}

## 6. Candidate evaluation
candidate wins 7 (robust 7); offgate emissions 0; regressions 0.

## 7. Attribution
{'NO_DELTA': 17, 'GATE_NO_FIRE': 16, 'BASELINE_DUPLICATE': 15, 'TRUE_FLI3_DELTA': 7}

## 8. Safety / offgate / determinism
verdict FLI3_SAFE; offgate 0; regressions 0; vacuous 0.

## 9. Discovery → validation comparison
FLI2 discovered 6 robust rescues; under literal-RC2 additive validation 6 reproduce as TRUE_FLI3_DELTA (all). Family holdout produced 1 additional generalization wins, so failure-derived discovery DOES generalize beyond the exact discovered theorems. Failure-derived discovery CAN feed RC-style validation.

## 10. Best validated examples

### `List.bidirectionalRec_nil` (family_holdout, LIST_DEF_UNFOLD)
- deploy `List.bidirectionalRec` via `simp [List.bidirectionalRec]`; literal RC2 failed; gate fired (LIST_DEF_UNFOLD); `simp [List.bidirectionalRec]` deploying `List.bidirectionalRec` solves; controls [] failed; robust; non-vacuous

### `Finset.card_le_one_iff` (rescue_replay, FINSET_CARD_BRIDGE)
- deploy `Finset.card_le_one` via `simp [Finset.card_le_one] <;> aesop`; literal RC2 failed; gate fired (FINSET_CARD_BRIDGE); `simp [Finset.card_le_one] <;> aesop` deploying `Finset.card_le_one` solves; controls [] failed; robust; non-vacuous

### `Finset.card_subtype` (rescue_replay, FINSET_MEM_DEF_UNFOLD)
- deploy `Finset.subtype` via `simp [Finset.subtype]`; literal RC2 failed; gate fired (FINSET_MEM_DEF_UNFOLD); `simp [Finset.subtype]` deploying `Finset.subtype` solves; controls [] failed; robust; non-vacuous

### `Finset.mem_filterMap` (rescue_replay, FINSET_MEM_DEF_UNFOLD)
- deploy `Finset.filterMap` via `simp [Finset.filterMap]`; literal RC2 failed; gate fired (FINSET_MEM_DEF_UNFOLD); `simp [Finset.filterMap]` deploying `Finset.filterMap` solves; controls [] failed; robust; non-vacuous

### `Finset.mem_map` (rescue_replay, FINSET_MEM_DEF_UNFOLD)
- deploy `Finset.map` via `simp [Finset.map]`; literal RC2 failed; gate fired (FINSET_MEM_DEF_UNFOLD); `simp [Finset.map]` deploying `Finset.map` solves; controls [] failed; robust; non-vacuous

### `Finset.mem_preimage` (rescue_replay, FINSET_MEM_DEF_UNFOLD)
- deploy `Finset.preimage` via `simp [Finset.preimage]`; literal RC2 failed; gate fired (FINSET_MEM_DEF_UNFOLD); `simp [Finset.preimage]` deploying `Finset.preimage` solves; controls [] failed; robust; non-vacuous

### `List.bidirectionalRec_singleton` (rescue_replay, LIST_DEF_UNFOLD)
- deploy `List.bidirectionalRec` via `simp [List.bidirectionalRec]`; literal RC2 failed; gate fired (LIST_DEF_UNFOLD); `simp [List.bidirectionalRec]` deploying `List.bidirectionalRec` solves; controls [] failed; robust; non-vacuous

## 11. Rejected or fragile examples

- control-duplicates: 0; flakes: 0; unknown-name/import gaps: 0.

## 12. Recommended FLI4

- Push validated families through full RC4-style floor benchmark + schema wrapper run; tighten gates for family generalization; address import-gap cases.
