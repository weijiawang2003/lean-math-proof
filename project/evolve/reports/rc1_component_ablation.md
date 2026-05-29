# RC1 component ablation

RC1 = NS9 ⊕ WX3 (Multiset induction) ⊕ MX2 (narrow Set.Finite aesop). The two additions act on disjoint namespaces and are additive to the NS9 ranked list, so each contribution is isolatable.

| component | Multiset surface | Set.Finite surface |
|---|---|---|
| NS9 only | 22 | 3 |
| NS9 + WX3 | 34 (+12) | 3 |
| NS9 + MX2 | 22 | 6 (+3) |
| **RC1 (NS9+WX3+MX2)** | **34 (+12)** | **6 (+3)** |

- **WX3 contributes +12 Multiset wins** (induction_on oracle).
- **MX2 contributes +3 Set.Finite wins** (narrow aesop fallback).
- **RC1 total gain over NS9 = +15** on the gain surfaces.
- **No negative interaction**: disjoint namespace gates; RC1 gain = WX3 + MX2.

