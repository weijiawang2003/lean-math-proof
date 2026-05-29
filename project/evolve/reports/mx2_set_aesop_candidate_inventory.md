# MX2 Set-aesop candidate inventory

MX1 found 2 new Set wins beyond production, both `over_attributed_raw` (a plain `aesop` closes them). The Set route carries no aesop fallback (unlike Finset/NS21). This inventories the misses + similar Set lemmas a Set-gated aesop fallback might also catch.

## Known aesop wins (2)

- `Set.Finite.toFinset_insert`
- `Set.Finite.toFinset_offDiag`

## Candidate buckets (fresh Set frontier)

| name prefix | count | already mined (MX1) |
|---|---|---|
| `Set.Finite.toFinset*` | 6 | 4 |
| `Set.Finite.*` | 65 | 18 |
| `Set.image*` | 21 | 0 |
| `Set.preimage*` | 7 | 0 |

**Total candidates: 99** across 4 buckets.

The Set route (gen_v5_ns12_balanced) has no aesop fallback, unlike Finset (NS21). MX2 adds a Set-gated aesop fallback mirroring NS19's finset_aesop_only and tests whether it captures these without regressions.
