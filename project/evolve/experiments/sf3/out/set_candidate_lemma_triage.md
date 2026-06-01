# SF3 Set Candidate-Lemma Triage

- candidates: 0 | rejected as not-missing-lemma: 12
- policy: conservative; a failure is a missing-lemma candidate only if unproven by all probes AND official proof is not an existing-lemma rw-bridge. Mirrors the Multiset-singleton negative result.

## Candidates (honest missing-lemma candidates)
- **None.** Every Set failure is explained by a tactic / routing / search-depth gap with existing lemmas — no new bridge lemma is warranted.
## Rejected (NOT missing lemmas)

| theorem | verdict | reason |
|---|---|---|
| `Set.antitoneOn_iff_antitone` | not_missing_lemma | probe solved it (tactic_gap): `simp [Antitone, AntitoneOn]` |
| `Set.ssubset_singleton_iff` | not_missing_lemma | probe solved it (search_depth_gap): `rw [ssubset_iff_subset_ne, subset_singleton_iff_eq, o |
| `Set.ite_empty_right` | not_missing_lemma | probe solved it (tactic_gap): `simp [Set.ite]` |
| `Set.ite_right` | not_missing_lemma | probe solved it (tactic_gap): `simp [Set.ite]` |
| `Set.diff_singleton_subset_iff` | not_missing_lemma | probe solved it (search_depth_gap): `rw [← union_singleton, union_comm] <;> apply diff_sub |
| `Set.subset_insert_iff` | not_missing_lemma | official proof is an rw-bridge over EXISTING named lemmas (search-depth gap, not missing): |
| `Set.union_empty_iff` | not_missing_lemma | probe solved it (tactic_gap): `simp only [← subset_empty_iff, union_subset_iff]` |
| `Set.ite_inter` | not_missing_lemma | probe solved it (search_depth_gap): `rw [ite_inter_inter, ite_same]` |
| `Set.ite_inter_self` | not_missing_lemma | probe solved it (search_depth_gap): `rw [Set.ite, union_inter_distrib_right, diff_inter_se |
| `Set.ite_eq_of_subset_left` | not_missing_lemma | probe solved it (search_depth_gap): `ext x <;> by_cases hx : x ∈ t <;> simp [hx, Set.ite,  |
| `Set.subset_singleton_iff_eq` | not_missing_lemma | cluster gap is search_depth_gap, not a missing-lemma cluster |
| `Set.pair_eq_pair_iff` | not_missing_lemma | probe solved it (tactic_gap): `simp [subset_antisymm_iff, insert_subset_iff] <;> aesop` |