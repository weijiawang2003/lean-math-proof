# RC2 Release — Final Verification

- **overall_pass: True**

## 1. JSON validity
- `rc2_production_wrapper.json`: valid
- `rc2_component_summary.json`: valid
- `rc2_reproduction_config.json`: valid

## 2. Protected diff (RC1 wrapper + NS24 router)
- `(empty — RC1 & NS24 untouched)` → untouched = **True**

## 3. Canonical floors (frozen RC2 production wrapper)
| floor | RC2 solved | required | source | pass |
|---|---|---|---|---|
| demo_v1 | 12 | 11/15 | verify run1 (frozen RC2 wrapper, live) | True |
| nat_defs_medium | 37 | 37/38 | RC2==RC1 by construction (no Set.ite names) | True |
| nat_defs_large_v5 | 49 | 49/65 | RC2==RC1 by construction (no Set.ite names) | True |
- floors_pass = **True**

## 4. SET_ITE known wins (+5 over literal RC1)
- solved 5/5 = **True** (verify run1 (frozen RC2 production wrapper, live))
- wins: ['Set.ite_empty_right', 'Set.ite_right', 'Set.ite_empty', 'Set.ite_empty_left', 'Set.ite_left']

## 5. Determinism smoke (known_wins x2, frozen wrapper)
- run1 hash `bbbd688b72d00c06` == run2 hash `bbbd688b72d00c06` → deterministic = **True**, diffs=[]

## Reused vs rerun
- rerun live (frozen wrapper): ['demo_v1', 'set_ite_known_wins (x2)']
- reused by construction: ['nat_defs_medium', 'nat_defs_large_v5 (RC2==RC1 on non-Set.ite; corroborated by full benchmark RC2 run)']

> Frozen rc2_production_wrapper.json verified live on demo_v1 + the 5 SET_ITE wins; Nat floors are RC2==RC1 by construction (gate denies action on non-Set.ite names) and match the RC1 baseline (37/38, 49/65). demo_v1 12/15 (>=11 floor; +1 vs RC1 baseline 11 is timing variance, not an RC2 effect — no Set.ite names on demo_v1).