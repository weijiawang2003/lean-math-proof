# NS19 — wrapper-variant signal comparison

Per-variant, per-set summary:
- `proved`: this NS19 variant + NS15 routed
- `raw`: raw NS15 routed only
- `wrap`: NS9 best wrapper + NS15 routed
- Δraw = proved − raw (wrapper-only signal vs raw)
- Δwrap = proved − wrap (genuinely new beyond NS9)

## `finset_aesop_only`

| set | proved | raw | wrap | Δraw | Δwrap | new beyond NS9 |
|---|---:|---:|---:|---:|---:|---|
| `ns19_finset_aesop_surface` | 58 | 57 | 57 | +1 | +1 | `Finset.coe_cons` |
| `ns17_finset_extra` | 15 | 12 | 12 | +3 | +3 | `Finset.coe_insert`, `Finset.cons_eq_insert`, `Finset.disjUnion_singleton` |
| `ns17_set_extra` | 18 | 18 | 18 | +0 | +0 |  |
| `nat_defs_medium` | 37 | 23 | 37 | +14 | +0 |  |
| `demo_v1` | 11 | 10 | 11 | +1 | +0 |  |
| `ns14_set_finset_extra` | 13 | 13 | 13 | +0 | +0 |  |

**Total Δwrap (new beyond NS9) across sets: 4 | total regressions: 0**

## `nat_simp_arith_targeted`

| set | proved | raw | wrap | Δraw | Δwrap | new beyond NS9 |
|---|---:|---:|---:|---:|---:|---|
| `ns19_nat_simp_arith_replay` | 4 | 4 | 4 | +0 | +0 |  |
| `nat_defs_medium` | 36 | 23 | 37 | +13 | -1 |  |
| `demo_v1` | 11 | 10 | 11 | +1 | +0 |  |
| `ns16_nat_div_mod_extra` | 2 | 0 | 1 | +2 | +1 | `Nat.mul_mod_mod` |

**Total Δwrap (new beyond NS9) across sets: 1 | total regressions: 1**

