# RC4 off-gate audit

- off-gate emissions (must be 0): **0** → **OFFGATE_CLEAN**
- component emission counts: {'RC4A': 76, 'RC4B': 18, 'RC4C_residue': 20}
- emitted-and-failed by component: {'RC4A': {'fired': 76, 'failed': 69, 'rate': 0.908}, 'RC4B': {'fired': 18, 'failed': 3, 'rate': 0.167}, 'RC4C_residue': {'fired': 20, 'failed': 8, 'rate': 0.4}}
- broad-gate warnings: ['RC4A emitted-and-failed rate 0.908 (fired 76)']

## Per-set gate emissions

| set | n | emissions | must_not_fire |
|---|---|---|---|
| canonical_demo_v1 | 15 | 0 | False |
| canonical_nat_defs_medium | 38 | 0 | False |
| canonical_nat_defs_large_v5 | 65 | 0 | False |
| rc4_known_wins | 23 | 23 | False |
| fresh_out_of_sample_frontier | 125 | 80 | False |
| negative_controls | 44 | 0 | True |
| offgate_controls | 45 | 0 | True |

## Emitted actions histogram

- def_unfold_simp_allowlist: 76
- MULTISET_DISJOINT_RIGHT_D2: 22
- LIST_FORALL_D2: 16
- MULTISET_DISJOINT_LEFT_SIMP: 11
- MULTISET_DISJOINT_LEFT_SIMP_AESOP: 11
- SET_DISJOINT_LEFT_SIMP: 7
- SET_DISJOINT_LEFT_SIMP_AESOP: 7
- SET_SUBSET_PAIR_D2: 2
