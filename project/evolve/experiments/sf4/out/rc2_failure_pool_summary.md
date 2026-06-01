# SF4 RC2 failure pool

- pool size (with file_path): **40**
- confirmed RC2 failures (from literal_rc2): **13**
- frontier unconfirmed candidates: **27**
- unresolved (no file_path): 0
- excluded (RC2 already solves): 17

## By namespace

- Set: 33
- Multiset: 3
- Eq: 1
- Function: 1
- GENERAL_FRONTIER: 1
- Prop: 1

## Confirmed RC2 failures (priority pool)

- `Multiset.toFinset_eq_singleton_iff` (Mathlib/Data/Finset/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.antitoneOn_iff_antitone` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.diff_singleton_subset_iff` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.ite_eq_of_subset_left` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.ite_eq_of_subset_right` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.ite_inter_of_inter_eq` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.pair_eq_pair_iff` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.powerset_singleton` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.ssubset_singleton_iff` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.subset_insert_iff` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.subset_ite` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.subset_singleton_iff_eq` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf
- `Set.union_empty_iff` (Mathlib/Data/Set/Basic.lean) — err: omega could not prove the goal:
No usable constraints found. You may need to unf

## Excluded (RC2 already solves / production-subsumed)

- `Set.diff_union_inter`
- `Set.insert_diff_eq_singleton`
- `Set.insert_diff_of_mem`
- `Set.ite_compl`
- `Set.ite_inter`
- `Set.ite_inter_compl_self`
- `Set.ite_inter_inter`
- `Set.ite_inter_self`
- `Set.ite_univ`
- `Set.mem_dite`
- `Set.mem_dite_empty_left`
- `Set.mem_dite_empty_right`
- `Set.mem_dite_univ_left`
- `Set.mem_dite_univ_right`
- `Set.mem_ite_empty_left`
- `Set.mem_ite_empty_right`
- `Set.pair_diff_left`