# WX3 — preservation matrix

WX3 configs = `ns9_best_genome.json` + a `Multiset.`-gated symbolic block. WX3 base is **byte-identical** to the NS9 genome, and every Multiset action is namespace-gated, so on non-Multiset theorems the ranked tactic list is identical to NS9 — preservation by construction. Empirical confirmation:

| set | ns class | NS9 | WX3-comb | Δ | regress | Multiset emit | floor |
|---|---|---:|---:|---:|---:|---:|---|
| demo_v1 | mixed-Nat/Set/Finset | 11 | 11 | +0 | 0 | 0 | 11/15 |
| nat_defs_medium | Nat | 37 | 37 | +0 | 0 | 0 | 37/38 |
| ns17_set_extra | Set | 18 | 18 | +0 | 0 | 0 | — |
| ns17_finset_extra | Finset | 15 | 15 | +0 | 0 | 0 | — |

By-construction (ranked-list identity; not re-run):
- `nat_defs_large_v5` (Nat (49/65 floor))
- `ns14_set_finset_extra` (Set/Finset)

- **Total regressions vs NS9: 0.**
- **Multiset symbolic emissions outside Multiset: 0.** (Namespace gate holds.)
- NS9 canonical floors preserved: medium 37/38, large 49/65, demo 11/15.
