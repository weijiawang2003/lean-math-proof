# RC4 release wrapper diff vs RC2

- verdict: **WRAPPER_DIFF_CLEAN**
- added tactics (15): prepended to `priority_templates['any']`
- added gates: 15 | added top-level keys: ['_rc4_release_candidate_metadata']
- modified fields (intended = priority_templates, theorem_name_tactic_gates): ['priority_templates', 'theorem_name_tactic_gates']
- removed keys: [] | unrelated changes: []
- RC2 priority.any preserved: True | RC2 gates preserved: True

## Component mapping

| component | tactics |
|---|---|
| RC4A | ['simp [Finset.disjUnion]', 'simp [Monotone, MonotoneOn]', 'simp [Antitone, AntitoneOn]', 'simp [StrictMono, StrictMonoOn]', 'simp [StrictAnti, StrictAntiOn]'] |
| RC4B | ['simp [Set.disjoint_left]', 'simp [Set.disjoint_left] <;> aesop', 'simp [Multiset.disjoint_left]', 'simp [Multiset.disjoint_left] <;> aesop'] |
| RC4C_residue | ['simp [Multiset.disjoint_right]', 'simp [Multiset.disjoint_right] <;> aesop', 'simp [Set.subset_pair_iff_eq]', 'simp [Set.subset_pair_iff_eq] <;> aesop', 'simp [List.forall_iff_forall_mem]', 'simp [List.forall_iff_forall_mem] <;> aesop'] |

- unchanged RC2 fields (27): ['_rc1_note', '_rc2_release_metadata', 'fallback_tactics', 'family_budgets', 'max_extra_tactics_per_state', 'max_steps', 'priority_template_budget', 'retrieval_enabled', 'retrieval_family_gates', 'retrieval_filter_self', 'retrieval_filter_unavailable', 'retrieval_requires_family', 'retrieval_shape_filter', 'retrieval_skip_bloating_apply', 'retrieval_tactic_forms', 'retrieval_top_k', 'symbolic_actions', 'symbolic_predictor', 'symbolic_sequence_search', 'tactic_templates', 'term_builder_budget', 'term_builder_templates', 'theorem_family_tactics', 'theorem_tactic_denylist', 'timeout_per_theorem', 'top_k', 'use_skeleton_bag']
