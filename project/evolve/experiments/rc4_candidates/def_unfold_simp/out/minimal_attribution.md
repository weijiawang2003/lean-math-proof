# RC4A minimal attribution

- new wins examined: 5
- classifications: {'TRUE_DEF_UNFOLD_SIMP_WIN': 5}
- **TRUE_DEF_UNFOLD_SIMP_WIN: 5** ['Finset.mem_disjUnion', 'Set.antitoneOn_iff_antitone', 'Set.monotoneOn_iff_monotone', 'Set.strictAntiOn_iff_strictAnti', 'Set.strictMonoOn_iff_strictMono']

| theorem | candidate | controls_solved | cand_solved | class |
|---|---|---|---|---|
| `Finset.mem_disjUnion` | `simp [Finset.disjUnion]` | [] | True | TRUE_DEF_UNFOLD_SIMP_WIN |
| `Set.antitoneOn_iff_antitone` | `simp [Antitone, AntitoneOn]` | [] | True | TRUE_DEF_UNFOLD_SIMP_WIN |
| `Set.monotoneOn_iff_monotone` | `simp [Monotone, MonotoneOn]` | [] | True | TRUE_DEF_UNFOLD_SIMP_WIN |
| `Set.strictAntiOn_iff_strictAnti` | `simp [StrictAnti, StrictAntiOn]` | [] | True | TRUE_DEF_UNFOLD_SIMP_WIN |
| `Set.strictMonoOn_iff_strictMono` | `simp [StrictMono, StrictMonoOn]` | [] | True | TRUE_DEF_UNFOLD_SIMP_WIN |
