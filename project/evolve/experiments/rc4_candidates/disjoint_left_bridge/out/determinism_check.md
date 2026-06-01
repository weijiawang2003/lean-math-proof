# RC4B determinism check

- targets: 74 (39 gate-firing)
- clean run1 hash: `7574d704d3505a47` | clean run2 hash: `7574d704d3505a47`
- gate decisions stable (all targets): **True**
- genuine diffs (clean theorems): **0** | flake-induced diffs: 0 | open flakes: 5
- **deterministic (modulo infrastructure flakes): True**

hash computed over cleanly-executed theorems; open flakes are Dojo hard-timeout / worker-kill infrastructure events on heavy-aesop / hard-Set goals, excluded from the hash. deterministic=True ⇔ identical gate decisions + identical solved outcomes on every cleanly-executed theorem.

## Open flakes (5) — infrastructure, excluded from hash

- Set._root_.Disjoint.image
- Set.biUnion_compl_eq_of_pairwise_disjoint_of_iUnion_eq_univ
- Set.disjoint_iUnion
- Multiset.coe_disjoint
- Multiset.disjoint_of_subset_left
