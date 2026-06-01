# TR7 gate refinement analysis

- candidate proposals: ['RC4A_TIGHTEN_MONO_GATE', 'RC4B_KEEP', 'RC4C_RESIDUE_KEEP', 'DYNAMIC_RETRIEVAL_REQUIRED']
- RC4A broad gate: fired 76, closed 7, precision 0.092 → **TIGHTEN**
- missing wins needing dynamic/new-lemma: 6

| component | fired | closed | precision | missed TR6 | change |
|---|---|---|---|---|---|
| RC4A | 76 | 7 | 0.092 | 0 | **tighten** |
| RC4B | 18 | 15 | 0.833 | 4 | **keep** |
| RC4C_residue | 20 | 12 | 0.6 | 0 | **keep** |

## Rationale

- **RC4A** (tighten): def-unfold gate fires 76× but closes only 7 (precision 0.092); it fires on every monotone/antitone theorem whether or not `simp [Monotone,…]` finishes — broad, low precision. Tighten to the iff-unfold shape (e.g. require `_iff_` in the name) or make it dynamic. Additive/safe today, but the loosest component.
- **RC4B** (keep): tight gate (precision 0.833); fires 18× closes 15. Covers all its TR6 disjoint wins. The 1 wrapper miss (Set.disjoint_sUnion_right) is a search-depth gap, not a gate problem.
- **RC4C_residue** (keep): tight gate (precision 0.6); fires 20× closes 12. Covers its TR6 residue wins. Missing TR6 wins are single-occurrence / theorem-specific (Part 6), so allowlist expansion is not yet warranted.
