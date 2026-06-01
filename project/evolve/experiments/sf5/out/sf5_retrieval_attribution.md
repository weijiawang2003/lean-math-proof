# SF5 retrieval attribution

- targets: 20
- classification: {'PROOF_DEPTH_GAP': 15, 'RETRIEVAL_ROUTING_GAP': 1, 'EXISTING_LEMMA_GAP': 4}
- retrieval wins over literal RC2: **5** (existing-lemma 4, routing 1)
- TRUE_MISSING_BRIDGE_LEMMA: **0**, PROOF_DEPTH_GAP: 15

| target | class | winning_lemma | evidence |
|---|---|---|---|
| Multiset.toFinset_eq_singleton_iff | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~12 steps, first `refine ⟨fun H ↦ ⟨fun h ↦ |
| Set.Nonempty.subset_pair_iff_eq | RETRIEVAL_ROUTING_GAP | Set.subset_pair_iff_eq | `aesop (add simp [Set.subset_pair_iff_eq])` closes via library search / hinted a |
| Set.antitoneOn_iff_antitone | EXISTING_LEMMA_GAP | Antitone+AntitoneOn | `simp [Antitone, AntitoneOn]` closes (existing Mathlib lemma, generic) |
| Set.diff_singleton_subset_iff | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~2 steps, first `rw [← union_singleton, un |
| Set.ite_eq_of_subset_left | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~3 steps, first `ext x`); no single retrie |
| Set.ite_eq_of_subset_right | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~3 steps, first `ext x`); no single retrie |
| Set.monotoneOn_iff_monotone | EXISTING_LEMMA_GAP | Monotone+MonotoneOn | `simp [Monotone, MonotoneOn]` closes (existing Mathlib lemma, generic) |
| Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~2 steps, first `simp [monotoneOn_iff_mono |
| Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~2 steps, first `simp [monotoneOn_iff_mono |
| Set.pair_eq_pair_iff | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~2 steps, first `simp [subset_antisymm_iff |
| Set.ssubset_iff_insert | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~2 steps, first `simp only [insert_subset_ |
| Set.ssubset_iff_sdiff_singleton | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~2 steps, first `simp [ssubset_iff_insert, |
| Set.ssubset_singleton_iff | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~3 steps, first `rw [ssubset_iff_subset_ne |
| Set.strictAntiOn_iff_strictAnti | EXISTING_LEMMA_GAP | StrictAnti+StrictAntiOn | `simp [StrictAnti, StrictAntiOn]` closes (existing Mathlib lemma, generic) |
| Set.strictMonoOn_iff_strictMono | EXISTING_LEMMA_GAP | StrictMono+StrictMonoOn | `simp [StrictMono, StrictMonoOn]` closes (existing Mathlib lemma, generic) |
| Set.subset_insert_iff | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~4 steps, first `rw [← diff_singleton_subs |
| Set.subset_ite | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~4 steps, first `simp only [subset_def, ←  |
| Set.subset_pair_iff_eq | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~6 steps, first `refine ⟨?_, by rintro (rf |
| Set.subset_singleton_iff_eq | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~3 steps, first `obtain rfl \| hs := s.eq_ |
| Set.union_empty_iff | PROOF_DEPTH_GAP |  | existing Mathlib proof is multi-step (~2 steps, first `simp only [← subset_empty |
