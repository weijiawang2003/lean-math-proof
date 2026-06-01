# RC4A — def_unfold_simp known wins

- known wins: **5**
- subfamilies: {'finset_def_unfold': 1, 'order_predicate_def_unfold': 4}
- validated def allowlist (9): ['Antitone', 'AntitoneOn', 'Finset.disjUnion', 'Monotone', 'MonotoneOn', 'StrictAnti', 'StrictAntiOn', 'StrictMono', 'StrictMonoOn']

| theorem | namespace | unfolded defs | subfamily |
|---|---|---|---|
| `Finset.mem_disjUnion` | Finset | Finset.disjUnion | finset_def_unfold |
| `Set.antitoneOn_iff_antitone` | Set | Antitone, AntitoneOn | order_predicate_def_unfold |
| `Set.monotoneOn_iff_monotone` | Set | Monotone, MonotoneOn | order_predicate_def_unfold |
| `Set.strictAntiOn_iff_strictAnti` | Set | StrictAnti, StrictAntiOn | order_predicate_def_unfold |
| `Set.strictMonoOn_iff_strictMono` | Set | StrictMono, StrictMonoOn | order_predicate_def_unfold |

## Mechanism

simp [<definitions named in the goal, restricted to the allowlist>]

All wins share one mechanism — definitional unfold via `simp [def]` where the def is named in the goal and is not @[simp]. Subfamilies differ only in WHICH defs (order predicates vs Finset.disjUnion), not in mechanism, so a single allowlist-gated action covers them; NOT CANDIDATE_TOO_HETEROGENEOUS.
