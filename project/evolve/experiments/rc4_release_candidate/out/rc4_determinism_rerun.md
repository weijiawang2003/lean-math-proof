# RC4 determinism rerun

- targets: 63 | compared: 63
- clean run1 hash: `ded0e256b75a12eb` | clean run2 hash: `ded0e256b75a12eb`
- genuine diffs: **0** | open flakes: 2 | win-affecting flakes: 0
- **deterministic (modulo infra flakes): True**

hash over cleanly-executed theorems; open flakes are Dojo hard-timeout / worker-kill infra events, excluded from the hash.

## Open flakes (2)

- Finset._root_.Monotone.map_finset_max'
- Finset._root_.Monotone.map_finset_min'
