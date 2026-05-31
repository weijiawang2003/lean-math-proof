# RC3 preservation + off-gate

- all floors pass: **True**
- any regression vs RC2 doc: **False**
- total off-gate emissions: **0** (expected 0)

## Canonical floors

| floor | RC3 solved | total | RC2 doc | floor min | pass | regression | off-gate emissions |
|---|---|---|---|---|---|---|---|
| demo_v1 | 12 | 15 | 12 | 11 | ✅ | 0 | 0 |
| nat_defs_medium | 37 | 38 | 37 | 37 | ✅ | 0 | 0 |
| nat_defs_large_v5 | 49 | 65 | 49 | 49 | ✅ | 0 | 0 |

## Negative controls

| theorem | available | finished |
|---|---|---|
| `Multiset.toFinset_eq_singleton_iff` | True | False |

Negative-control off-gate emissions: **0**

## Off-gate detail
Off-gate = SX3 sequence `simp [Set.ite] <;> aesop` emitted on a theorem whose name lacks `Set.ite`.

- **demo_v1**: on_gate=0 off_gate=0 
- **nat_defs_medium**: on_gate=0 off_gate=0 
- **nat_defs_large_v5**: on_gate=0 off_gate=0 