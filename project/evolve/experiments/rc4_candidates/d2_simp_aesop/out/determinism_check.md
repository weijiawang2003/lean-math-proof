# RC4C determinism check

- targets: 80 (42 gate-firing)
- clean run1 hash: `620a7e9d5dcdf044` | clean run2 hash: `620a7e9d5dcdf044`
- gate decisions stable (all targets): **True**
- genuine diffs (clean theorems): **0** | flake-induced diffs: 2 | open flakes: 4
- **deterministic (modulo infrastructure flakes): True**

hash computed over cleanly-executed theorems; open flakes are Dojo hard-timeout / worker-kill infrastructure events on heavy `<;> aesop` goals, excluded from the hash.

## Flake-induced diffs (excluded)

- `Multiset.singleton_disjoint`: run1 solved=True / run2 solved=False
- `Multiset.zero_disjoint`: run1 solved=True / run2 solved=False

## Open flakes (4) — infrastructure, excluded from hash

- List.filterMap_eq_map_iff_forall_eq_some
- Multiset.singleton_disjoint  ⚠ credited-win
- Multiset.zero_disjoint  ⚠ credited-win
- Set._root_.Disjoint.image
