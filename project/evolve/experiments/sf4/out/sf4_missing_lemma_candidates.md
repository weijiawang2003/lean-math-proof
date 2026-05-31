# SF4 missing-lemma triage

- clusters triaged: **10**
- histogram: {'POSSIBLE_MISSING_BRIDGE_LEMMA': 2, 'NEEDS_MORE_DATA': 7, 'PROOF_SEARCH_DEPTH_GAP': 1}

| cluster | ns | shape | size | category |
|---|---|---|---|---|
| `Set__iff__iff` | Set | iff | 16 | **POSSIBLE_MISSING_BRIDGE_LEMMA** |
| `Set__ite_if__subset` | Set | subset | 3 | **POSSIBLE_MISSING_BRIDGE_LEMMA** |
| `Multiset__iff__iff` | Multiset | iff | 1 | **NEEDS_MORE_DATA** |
| `Set__ite_if__equality` | Set | equality | 1 | **NEEDS_MORE_DATA** |
| `Set__singleton__arithmetic` | Set | arithmetic | 1 | **NEEDS_MORE_DATA** |
| `unknown__subset__subset` | unknown | subset | 1 | **NEEDS_MORE_DATA** |
| `unknown__iff__iff` | unknown | iff | 1 | **NEEDS_MORE_DATA** |
| `unknown__compl__arithmetic` | unknown | arithmetic | 1 | **PROOF_SEARCH_DEPTH_GAP** |
| `Set__other__equality` | Set | equality | 1 | **NEEDS_MORE_DATA** |
| `Set__map_filter__arithmetic` | Set | arithmetic | 1 | **NEEDS_MORE_DATA** |

## Detail

### `Set__iff__iff` — POSSIBLE_MISSING_BRIDGE_LEMMA
- members: ['Set.antitoneOn_iff_antitone', 'Set.diff_singleton_subset_iff', 'Set.pair_eq_pair_iff', 'Set.ssubset_singleton_iff', 'Set.subset_insert_iff', 'Set.subset_singleton_iff_eq', 'Set.union_empty_iff', 'Set.Nonempty.subset_pair_iff_eq', 'Set.monotoneOn_iff_monotone', 'Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le', 'Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt', 'Set.ssubset_iff_insert', 'Set.ssubset_iff_sdiff_singleton', 'Set.strictAntiOn_iff_strictAnti', 'Set.strictMonoOn_iff_strictMono', 'Set.subset_pair_iff_eq']
- rationale: repeated goal shape with NO generic tactic/sequence closing any member -> reusable bridge-lemma candidate (verify a Mathlib lemma does not already exist)

### `Set__ite_if__subset` — POSSIBLE_MISSING_BRIDGE_LEMMA
- members: ['Set.ite_eq_of_subset_left', 'Set.ite_eq_of_subset_right', 'Set.subset_ite']
- rationale: repeated goal shape with NO generic tactic/sequence closing any member -> reusable bridge-lemma candidate (verify a Mathlib lemma does not already exist)

### `Multiset__iff__iff` — NEEDS_MORE_DATA
- members: ['Multiset.toFinset_eq_singleton_iff']
- rationale: single unresolved theorem; insufficient repetition to infer a lemma

### `Set__ite_if__equality` — NEEDS_MORE_DATA
- members: ['Set.ite_inter_of_inter_eq']
- rationale: single unresolved theorem; insufficient repetition to infer a lemma

### `Set__singleton__arithmetic` — NEEDS_MORE_DATA
- members: ['Set.powerset_singleton']
- rationale: single unresolved theorem; insufficient repetition to infer a lemma

### `unknown__subset__subset` — NEEDS_MORE_DATA
- members: ['Eq.subset']
- rationale: single unresolved theorem; insufficient repetition to infer a lemma

### `unknown__iff__iff` — NEEDS_MORE_DATA
- members: ['Function.Injective.nonempty_apply_iff']
- rationale: single unresolved theorem; insufficient repetition to infer a lemma

### `unknown__compl__arithmetic` — PROOF_SEARCH_DEPTH_GAP
- members: ['Prop.compl_singleton']
- rationale: every member is closed in isolation by a bare control/probe but literal RC2 search did not reach it — depth/ordering gap, not a missing lemma

### `Set__other__equality` — NEEDS_MORE_DATA
- members: ['Set.eq_of_inclusion_surjective']
- rationale: single unresolved theorem; insufficient repetition to infer a lemma

### `Set__map_filter__arithmetic` — NEEDS_MORE_DATA
- members: ['Set.pairwiseDisjoint_filter']
- rationale: single unresolved theorem; insufficient repetition to infer a lemma


> Candidate directions only — no lemmas invented. Verify existence in Mathlib before any SF5 synthesis.