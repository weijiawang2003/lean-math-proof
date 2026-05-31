# SF2 Multiset seed — source context

## `Multiset.toFinset_nsmul`

- file: `Mathlib/Data/Finset/Basic.lean`  | source_found: `True`  | decl line: 3125
- nearby keyword lemmas: ['Finset.union_symm_inl', 'Finset.union_symm_inr', 'Nodup.toFinset_inj', 'Nonempty.exists_eq_singleton_or_nontrivial', 'Nonempty.subset_singleton_iff', 'Nontrivial.ne_singleton', 'Nontrivial.sdiff_singleton_nonempty', '_root_.Disjoint.forall_ne_finset', '_root_.Set.pairwiseDisjoint_filter', 'and', 'coe_disjUnion', 'coe_eq_singleton', 'coe_singleton', 'coe_subset_singleton', 'coe_toFinset', 'default_singleton', 'disjUnion_comm', 'disjUnion_empty', 'disjUnion_singleton', 'disjoint_coe', 'disjoint_empty_left', 'disjoint_empty_right', 'disjoint_erase_comm', 'disjoint_erase_insert', 'disjoint_filter', 'disjoint_filter_filter', "disjoint_filter_filter'", 'disjoint_filter_filter_neg', 'disjoint_iff_inter_eq_empty', 'disjoint_iff_ne', 'disjoint_insert_erase', 'disjoint_insert_left', 'disjoint_insert_right', 'disjoint_left', 'disjoint_of_erase_left', 'disjoint_of_erase_right', 'disjoint_of_subset_left', 'disjoint_of_subset_right', 'disjoint_or_nonempty_inter', 'disjoint_right']

Statement:
```lean
theorem toFinset_nsmul (s : Multiset α) : ∀ n ≠ 0, (n • s).toFinset = s.toFinset
  | 0, h => by contradiction
  | n + 1, _ => by
```
Existing Mathlib proof:
```lean
by_cases h : n = 0
    · rw [h, zero_add, one_nsmul]
    · rw [add_nsmul, toFinset_add, one_nsmul, toFinset_nsmul s n h, Finset.union_idempotent]
#align multiset.to_finset_nsmul Multiset.toFinset_nsmul

theorem toFinset_eq_singleton_iff (s : Multiset α) (a : α) :
    s.toFinset = {a} ↔ card s ≠ 0 ∧ s = card s • {a} := by
  refine ⟨fun H ↦ ⟨fun h ↦ ?_, ext' fun x ↦ ?_⟩, fun H ↦ ?_⟩
  · rw [card_eq_zero.1 h, toFinset_zero] at H
    exact Finset.singleton_ne_empty _ H.symm
  · rw [count_nsmul, count_singleton]
    by_cases hx : x = a
    · simp_rw [hx, ite_true, mul_one, count_eq_card]
      intro y hy
      rw [← mem_toFinset, H, Finset.mem_singleton] at hy
      exact hy.symm
    have hx' : x ∉ s := fun h' ↦ hx <| by rwa [← mem_toFinset, H, Finset.mem_singleton] at h'
    simp_rw [count_eq_zero_of_not_mem hx', hx, ite_false, Nat.mul_zero]
  simpa only [toFinset_nsmul _ _ H.1, toFinset_singleton] using congr($(H.2).toFinset)
```

## `Multiset.toFinset_eq_singleton_iff`

- file: `Mathlib/Data/Finset/Basic.lean`  | source_found: `True`  | decl line: 3133
- nearby keyword lemmas: ['Finset.union_symm_inl', 'Finset.union_symm_inr', 'Nodup.toFinset_inj', 'Nonempty.exists_eq_singleton_or_nontrivial', 'Nonempty.subset_singleton_iff', 'Nontrivial.ne_singleton', 'Nontrivial.sdiff_singleton_nonempty', '_root_.Disjoint.forall_ne_finset', '_root_.Set.pairwiseDisjoint_filter', 'and', 'coe_disjUnion', 'coe_eq_singleton', 'coe_singleton', 'coe_subset_singleton', 'coe_toFinset', 'default_singleton', 'disjUnion_comm', 'disjUnion_empty', 'disjUnion_singleton', 'disjoint_coe', 'disjoint_empty_left', 'disjoint_empty_right', 'disjoint_erase_comm', 'disjoint_erase_insert', 'disjoint_filter', 'disjoint_filter_filter', "disjoint_filter_filter'", 'disjoint_filter_filter_neg', 'disjoint_iff_inter_eq_empty', 'disjoint_iff_ne', 'disjoint_insert_erase', 'disjoint_insert_left', 'disjoint_insert_right', 'disjoint_left', 'disjoint_of_erase_left', 'disjoint_of_erase_right', 'disjoint_of_subset_left', 'disjoint_of_subset_right', 'disjoint_or_nonempty_inter', 'disjoint_right']

Statement:
```lean
theorem toFinset_eq_singleton_iff (s : Multiset α) (a : α) :
    s.toFinset = {a} ↔ card s ≠ 0 ∧ s = card s • {a} := by
```
Existing Mathlib proof:
```lean
refine ⟨fun H ↦ ⟨fun h ↦ ?_, ext' fun x ↦ ?_⟩, fun H ↦ ?_⟩
  · rw [card_eq_zero.1 h, toFinset_zero] at H
    exact Finset.singleton_ne_empty _ H.symm
  · rw [count_nsmul, count_singleton]
    by_cases hx : x = a
    · simp_rw [hx, ite_true, mul_one, count_eq_card]
      intro y hy
      rw [← mem_toFinset, H, Finset.mem_singleton] at hy
      exact hy.symm
    have hx' : x ∉ s := fun h' ↦ hx <| by rwa [← mem_toFinset, H, Finset.mem_singleton] at h'
    simp_rw [count_eq_zero_of_not_mem hx', hx, ite_false, Nat.mul_zero]
  simpa only [toFinset_nsmul _ _ H.1, toFinset_singleton] using congr($(H.2).toFinset)
```

## `Multiset.disjoint_toFinset`

- file: `Mathlib/Data/Finset/Basic.lean`  | source_found: `True`  | decl line: 3523
- nearby keyword lemmas: ['Finset.union_symm_inl', 'Finset.union_symm_inr', 'Nodup.toFinset_inj', 'Nonempty.exists_eq_singleton_or_nontrivial', 'Nonempty.subset_singleton_iff', 'Nontrivial.ne_singleton', 'Nontrivial.sdiff_singleton_nonempty', '_root_.Disjoint.forall_ne_finset', '_root_.Set.pairwiseDisjoint_filter', 'and', 'coe_disjUnion', 'coe_eq_singleton', 'coe_singleton', 'coe_subset_singleton', 'coe_toFinset', 'default_singleton', 'disjUnion_comm', 'disjUnion_empty', 'disjUnion_singleton', 'disjoint_coe', 'disjoint_empty_left', 'disjoint_empty_right', 'disjoint_erase_comm', 'disjoint_erase_insert', 'disjoint_filter', 'disjoint_filter_filter', "disjoint_filter_filter'", 'disjoint_filter_filter_neg', 'disjoint_iff_inter_eq_empty', 'disjoint_iff_ne', 'disjoint_insert_erase', 'disjoint_insert_left', 'disjoint_insert_right', 'disjoint_left', 'disjoint_of_erase_left', 'disjoint_of_erase_right', 'disjoint_of_subset_left', 'disjoint_of_subset_right', 'disjoint_or_nonempty_inter', 'disjoint_right']

Statement:
```lean
theorem disjoint_toFinset {m1 m2 : Multiset α} :
    _root_.Disjoint m1.toFinset m2.toFinset ↔ m1.Disjoint m2 := by
```
Existing Mathlib proof:
```lean
rw [Finset.disjoint_iff_ne]
  refine ⟨fun h a ha1 ha2 => ?_, ?_⟩
  · rw [← Multiset.mem_toFinset] at ha1 ha2
    exact h _ ha1 _ ha2 rfl
  · rintro h a ha b hb rfl
    rw [Multiset.mem_toFinset] at ha hb
    exact h ha hb
#align multiset.disjoint_to_finset Multiset.disjoint_toFinset

end Multiset

namespace List

variable [DecidableEq α] {l l' : List α}

theorem disjoint_toFinset_iff_disjoint : _root_.Disjoint l.toFinset l'.toFinset ↔ l.Disjoint l' :=
  Multiset.disjoint_toFinset
#align list.disjoint_to_finset_iff_disjoint List.disjoint_toFinset_iff_disjoint

end List

namespace Mathlib.Meta
open Qq Lean Meta Finset

/-- Attempt to prove that a finset is nonempty using the `finsetNonempty` aesop rule-set.

You can add lemmas to the rule-set by tagging them with either:
* `aesop safe apply (rule_sets := [finsetNonempty])` if they are always a good idea to follow or
* `aesop unsafe apply (rule_sets := [finsetNonempty])` if they risk directing the search to a blind
  alley.
-/
def proveFinsetNonempty {u : Level} {α : Q(Type u)} (s : Q(Finset $α)) :
    MetaM (Option Q(Finset.Nonempty $s)) := do
  -- Aesop expects to operate on goals, so we're going to make a new goal.
  let goal ← Lean.Meta.mkFreshExprMVar q(Finset.Nonempty $s)
  let mvar := goal.mvarId!
  -- We want this to be fast, so use only the basic and `Finset.Nonempty`-specific rules.
  let rulesets ← Aesop.Frontend.getGlobalRuleSets #[`builtin, `finsetNonempty]
  let options : Aesop.Options' :=
    { terminal := true -- Fail if the new goal is not closed.
      generateScript := false
      useDefaultSimpSet := false -- Avoiding the whole simp set to speed up the tactic.
      warnOnNonterminal := false -- Don't show a warning on failure, simply return `none`.
      forwardMaxDepth? := none }
  let rules ← Aesop.mkLocalRuleSet rulesets options
  let (remainingGoals, _) ←
    try Aesop.search (options := options.toOptions) mvar (.some rules)
    catch _ => return none
  -- Fail if there are open goals remaining, this serves as an extra check for the
  -- Aesop configuration option `terminal := true`.
  if remainingGoals.size > 0 then return none
  Lean.getExprMVarAssignment? mvar

end Mathlib.Meta
```
