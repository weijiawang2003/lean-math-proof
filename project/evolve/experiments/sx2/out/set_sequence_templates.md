# SX2 — Mined Set Proof-Sequence Templates

- winning probes mined: 10 | template families: 4 | interesting (gate-worthy): 1
- A template is gate-worthy only if it solves >=2 theorems or is a pure tactic reusable by goal shape with NO theorem-specific symbols. Mined evidence: only SET_ITE_SIMP qualifies (n=2, simp [Set.ite]); all rw-bridges / simp-sets are single-theorem and theorem-specific.

## Template support

| family | n | theorems | named-lemmas | local-hyp | gen-risk | source-copy | interesting |
|---|---|---|---|---|---|---|---|
| `SET_RW_BRIDGE` | 4 | diff_singleton_subset_iff, ite_inter, ite_inter_self, ssubset_singleton_iff | True | False | high | high | **False** (family solves 4 but each member is theorem-specific (named lemmas=True, local-hyp=False) — NO single reusable tactic, NOT gate-worthy) |
| `SOURCE_SPECIFIC` | 3 | antitoneOn_iff_antitone, pair_eq_pair_iff, union_empty_iff | True | False | high | high | **False** (family solves 3 but each member is theorem-specific (named lemmas=True, local-hyp=False) — NO single reusable tactic, NOT gate-worthy) |
| `SET_ITE_SIMP` | 2 | ite_empty_right, ite_right | False | False | low | low | **True** (theorem-agnostic tactic solves 2 theorems) |
| `SET_EXT_BYCASES` | 1 | ite_eq_of_subset_left | True | True | high | high | **False** (family solves 1 but each member is theorem-specific (named lemmas=True, local-hyp=True) — NO single reusable tactic, NOT gate-worthy) |

## Per-winning-probe detail

### `Set.diff_singleton_subset_iff` — `SET_RW_BRIDGE`
- winning tactic: `rw [← union_singleton, union_comm] <;> apply diff_subset_iff`
- normalized: `rw [<bridge lemmas>] (+ closer)` | shape: equality
- required symbols: ['union_comm', 'union_singleton']
- local hyps used: [] | gen-risk: high | theorem-specific-risk: high
- parser constraints: ['single_line_only', "';'->'<;>' rewrite", "no '·' bullet blocks"]

### `Set.ite_eq_of_subset_left` — `SET_EXT_BYCASES`
- winning tactic: `ext x <;> by_cases hx : x ∈ t <;> simp [hx, Set.ite, or_iff_right_of_imp (@h x)]`
- normalized: `ext x <;> by_cases <VAR> <;> simp_all [...]` | shape: equality
- required symbols: ['Set.ite', 'hx', 'or_iff_right_of_imp']
- local hyps used: ['@h', 'hx'] | gen-risk: high | theorem-specific-risk: high
- parser constraints: ['single_line_only', "';'->'<;>' rewrite", "no '·' bullet blocks"]

### `Set.pair_eq_pair_iff` — `SOURCE_SPECIFIC`
- winning tactic: `simp [subset_antisymm_iff, insert_subset_iff] <;> aesop`
- normalized: `simp [subset_antisymm_iff, insert_subset_iff] <;> aesop` | shape: equality
- required symbols: ['insert_subset_iff', 'subset_antisymm_iff']
- local hyps used: [] | gen-risk: high | theorem-specific-risk: high
- parser constraints: ['single_line_only', "';'->'<;>' rewrite", "no '·' bullet blocks"]

### `Set.union_empty_iff` — `SOURCE_SPECIFIC`
- winning tactic: `simp only [← subset_empty_iff, union_subset_iff]`
- normalized: `simp only [← subset_empty_iff, union_subset_iff]` | shape: equality
- required symbols: ['subset_empty_iff', 'union_subset_iff']
- local hyps used: [] | gen-risk: high | theorem-specific-risk: high
- parser constraints: ['single_line_only']

### `Set.antitoneOn_iff_antitone` — `SOURCE_SPECIFIC`
- winning tactic: `simp [Antitone, AntitoneOn]`
- normalized: `simp [Antitone, AntitoneOn]` | shape: iff
- required symbols: ['Antitone', 'AntitoneOn']
- local hyps used: [] | gen-risk: high | theorem-specific-risk: high
- parser constraints: ['single_line_only']

### `Set.ssubset_singleton_iff` — `SET_RW_BRIDGE`
- winning tactic: `rw [ssubset_iff_subset_ne, subset_singleton_iff_eq, or_and_right, and_not_self_iff, or_false_iff, and_iff_left_iff_imp] <;> exact fun h => h ▸ (singleton_ne_empty _).symm`
- normalized: `rw [<bridge lemmas>] (+ closer)` | shape: iff
- required symbols: ['and_iff_left_iff_imp', 'and_not_self_iff', 'or_and_right', 'or_false_iff', 'ssubset_iff_subset_ne', 'subset_singleton_iff_eq']
- local hyps used: [] | gen-risk: high | theorem-specific-risk: high
- parser constraints: ['single_line_only', "';'->'<;>' rewrite", "no '·' bullet blocks"]

### `Set.ite_empty_right` — `SET_ITE_SIMP`
- winning tactic: `simp [Set.ite]`
- normalized: `simp [Set.ite]` | shape: membership
- required symbols: ['Set.ite']
- local hyps used: [] | gen-risk: low | theorem-specific-risk: low
- parser constraints: ['single_line_only']

### `Set.ite_inter` — `SET_RW_BRIDGE`
- winning tactic: `rw [ite_inter_inter, ite_same]`
- normalized: `rw [<bridge lemmas>] (+ closer)` | shape: membership
- required symbols: ['ite_inter_inter', 'ite_same']
- local hyps used: [] | gen-risk: high | theorem-specific-risk: high
- parser constraints: ['single_line_only']

### `Set.ite_inter_self` — `SET_RW_BRIDGE`
- winning tactic: `rw [Set.ite, union_inter_distrib_right, diff_inter_self, inter_assoc, inter_self, union_empty]`
- normalized: `rw [<bridge lemmas>] (+ closer)` | shape: membership
- required symbols: ['Set.ite', 'diff_inter_self', 'inter_assoc', 'inter_self', 'union_empty', 'union_inter_distrib_right']
- local hyps used: [] | gen-risk: high | theorem-specific-risk: high
- parser constraints: ['single_line_only']

### `Set.ite_right` — `SET_ITE_SIMP`
- winning tactic: `simp [Set.ite]`
- normalized: `simp [Set.ite]` | shape: membership
- required symbols: ['Set.ite']
- local hyps used: [] | gen-risk: low | theorem-specific-risk: low
- parser constraints: ['single_line_only']
