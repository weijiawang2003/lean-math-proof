# TR7 distribution mismatch

- **verdict: PARTIAL_DISTRIBUTION_MISMATCH**
- TR6 fresh wins present in RC4R fresh frontier: **0/18**
- TR6 winning patterns represented in RC4R fresh: 100%
- RC4R fresh gate firing: 64%, component split {'RC4A': 71, 'RC4C_residue': 7, 'RC4B': 2}
- over-sampling: RC4R fresh fires RC4A 71× vs RC4B+RC4C 9×; TR6 fresh wins were 13/18 disjoint-shaped

## Namespace distribution

| namespace | TR6 batch | TR6 fresh wins | RC4R fresh |
|---|---|---|---|
| Finset | 30 | 2 | 16 |
| Set | 21 | 5 | 13 |
| Multiset | 22 | 9 | 8 |
| List | 29 | 1 | 8 |
| Monotone | 0 | 0 | 7 |
| Antitone | 0 | 0 | 7 |
| AntitoneOn | 2 | 0 | 7 |
| MonotoneOn | 2 | 0 | 7 |
| Nat | 14 | 1 | 6 |
| Function | 0 | 0 | 3 |
| BddBelow | 0 | 0 | 2 |
| BddAbove | 0 | 0 | 2 |
| Prop | 0 | 0 | 1 |
| and_iff_not_or_not | 0 | 0 | 1 |
| ULift | 0 | 0 | 1 |
| apply_dite | 0 | 0 | 1 |
| ExistsUnique | 0 | 0 | 1 |
| BEx | 0 | 0 | 1 |
| strictMono_restrict | 0 | 0 | 1 |
| PartialOrder | 0 | 0 | 1 |
| OrderDual | 0 | 0 | 1 |
| Subtype | 0 | 0 | 1 |
| LE | 0 | 0 | 1 |
| PUnit | 0 | 0 | 1 |
| IsGreatest | 0 | 0 | 1 |
| Iff | 0 | 0 | 1 |
| Equiv | 1 | 0 | 1 |
| Option | 3 | 0 | 1 |
| IsGLB | 1 | 0 | 1 |
| Bool | 0 | 0 | 1 |
| Eq | 0 | 0 | 1 |
| Int | 0 | 0 | 1 |
| LT | 0 | 0 | 1 |
| Decidable | 0 | 0 | 1 |
| Pi | 0 | 0 | 1 |
| ScottContinuous | 0 | 0 | 1 |
| Ne | 0 | 0 | 1 |
| IsLUB | 1 | 0 | 1 |
| and_forall_ne | 0 | 0 | 1 |
| Prod | 0 | 0 | 1 |
| and_symm_left | 0 | 0 | 1 |
| Exists | 0 | 0 | 1 |
| and_or_imp | 0 | 0 | 1 |
| Sum | 0 | 0 | 1 |
| LinearOrder | 0 | 0 | 1 |
| Preorder | 0 | 0 | 1 |
| IsLeast | 0 | 0 | 1 |
| PLift | 1 | 0 | 1 |
| and_symm_right | 0 | 0 | 1 |
| OrderBot | 0 | 0 | 1 |
| OrderTop | 0 | 0 | 1 |
| bddAbove_iff_subset_Iic | 1 | 0 | 0 |
| bddBelow_def | 1 | 0 | 0 |
| mem_lowerBounds_iff_subset_Ici | 1 | 0 | 0 |
| bddBelow_iff_subset_Ici | 1 | 0 | 0 |
| bddBelow_bddAbove_iff_subset_Icc | 1 | 0 | 0 |
| exists_of_exists_mem | 1 | 0 | 0 |
| isLeast_union_iff | 1 | 0 | 0 |
| isGreatest_union_iff | 1 | 0 | 0 |
| exists_mem_or | 1 | 0 | 0 |
| bddAbove_def | 1 | 0 | 0 |

## Feature profile (fraction of set)

| feature | TR6 batch | TR6 fresh wins | RC4R fresh |
|---|---|---|---|
| has_disjoint | 0.285 | 0.722 | 0.016 |
| has_subset | 0.277 | 0.111 | 0.088 |
| has_iff | 0.92 | 1.0 | 0.184 |
| has_mem | 0.401 | 0.5 | 0.152 |
| has_singleton | 0.584 | 0.889 | 0.408 |
| has_union_inter | 0.204 | 0.333 | 0.104 |
| has_map_filter | 0.423 | 0.167 | 0.376 |
| has_tofinset | 0.007 | 0.0 | 0.0 |
| has_nat_arith | 0.219 | 0.111 | 0.2 |
| has_order | 0.255 | 0.056 | 0.544 |
| has_eq | 0.358 | 0.222 | 0.312 |
| has_card | 0.182 | 0.167 | 0.096 |

## TR6 winning patterns missing from RC4R fresh

