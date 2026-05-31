# RC3 validation comparison

## DECISION: `REJECT_NO_LITERAL_DELTA`

## Headline

- validation surface: **30** theorems
- literal RC2 solved: **17**
- literal RC3 solved: **17**
- raw delta: **+0**
- **credited SX3 delta (minimal-attribution TRUE wins): 0**
  - reproduced deferred: []
  - fresh true wins: []
- regressions vs RC2: **0** 
- off-gate emissions: **0**
- canonical floors pass: **True**
- determinism: **deterministic** (hashes 5757bbb27215 vs 5757bbb27215, match=True, open_flakes=0)

## Criteria for RC3_RELEASE_CANDIDATE_CONFIRMED

- ❌ positive_credited_delta
- ❌ at_least_one_fresh_true_win
- ✅ zero_regressions
- ✅ zero_off_gate
- ✅ canonical_floors_pass
- ❌ minimal_attribution_confirms
- ✅ deterministic_or_env_flake_only

## New wins over literal RC2


## Canonical floors

| floor | RC3 solved | total | min | pass |
|---|---|---|---|---|
| demo_v1 | 12 | 15 | 11 | ✅ |
| nat_defs_medium | 37 | 38 | 37 | ✅ |
| nat_defs_large_v5 | 49 | 65 | 49 | ✅ |