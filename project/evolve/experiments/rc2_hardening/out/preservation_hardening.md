# RC2 Hardening — Preservation / Off-Gate

- **hardening_ok = True**
- speculative gates present: NONE
- off-gate emissions (non-Set surfaces): NONE | positive Set.ite controls fired: 2/2
- canonical floors pass: True — {'demo_v1': {'solved': 11, 'floor': '>=11/15', 'pass': True}, 'nat_defs_medium': {'solved': 37, 'floor': '>=37/38', 'pass': True}, 'nat_defs_large_v5': {'solved': 49, 'floor': '>=49/65', 'pass': True}}
- regressions: NONE

> Gate is name-prefixed to Set.ite; fires only on Set.ite* (off-gate=0 by construction). Speculative gates absent. Floors preserved (RC2==RC1 on non-Set.ite by construction). RC1/NS24 untouched.