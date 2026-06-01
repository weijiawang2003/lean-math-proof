/-
SF3 Set candidate-lemma attempts — INTENTIONALLY EMPTY OF CANDIDATES.

The SF2 Set-cluster deep dive triaged 12 RC1 Set failures and produced
ZERO honest missing-lemma candidates: every failure is an automation gap
(tactic / routing / search-depth) over EXISTING Mathlib lemmas, mirroring the
Multiset.toFinset_eq_singleton_iff negative result.

Evidence (live LeanDojo probes; see probe_results.json):
  • Set.ite_right, Set.ite_empty_right      closed by `simp [Set.ite]`  (tactic gap)
  • Set.antitoneOn_iff_antitone             closed by `simp [Antitone, AntitoneOn]`
  • Set.union_empty_iff                     closed by `simp only [← subset_empty_iff, union_subset_iff]`
  • Set.pair_eq_pair_iff                    closed by `simp [subset_antisymm_iff, insert_subset_iff] <;> aesop`
  • Set.ite_inter, Set.ite_inter_self,      closed by official rw-bridges over existing lemmas
    Set.diff_singleton_subset_iff,
    Set.ssubset_singleton_iff,
    Set.ite_eq_of_subset_left
  • Set.subset_insert_iff,                  proof exists (rw + by_cases over existing lemmas) but
    Set.subset_singleton_iff_eq             needs per-branch bullets that single-line probes can't carry

No new bridge lemma is warranted. Do NOT edit Mathlib or production code.
-/
