# SF2 Set Cluster — Source Context & Official-Proof Analysis

- cases: 12 | context lines: ±120
- proof-style histogram: `{'rw_bridge': 5, 'by_cases_ite_split': 1, 'subset_antisymm': 1, 'simp_only': 5}`

## `Set.diff_singleton_subset_iff`
- file: `Mathlib/Data/Set/Basic.lean` (line 1903)
- proof_style: **rw_bridge** | ext=False by_cases=False rw=True simp_set=False aesop=False
- likely_reusable_probe: `rw-bridge (theorem-specific; see official proof)`
- notes: rewrite-bridge proof: depends on specific named lemmas, not closed by generic simp/aesop

```lean
theorem diff_singleton_subset_iff {x : α} {s t : Set α} : s \ {x} ⊆ t ↔ s ⊆ insert x t := by
  rw [← union_singleton, union_comm]
  apply diff_subset_iff
```

## `Set.ite_eq_of_subset_left`
- file: `Mathlib/Data/Set/Basic.lean` (line 2344)
- proof_style: **by_cases_ite_split** | ext=True by_cases=True rw=False simp_set=True aesop=False
- likely_reusable_probe: `simp [*, Set.ite, or_iff_right_of_imp (@h x)]`
- notes: ext + by_cases on membership in t, then simp with hypotheses

```lean
theorem ite_eq_of_subset_left (t : Set α) {s₁ s₂ : Set α} (h : s₁ ⊆ s₂) :
    t.ite s₁ s₂ = s₁ ∪ (s₂ \ t) := by
  ext x
  by_cases hx : x ∈ t <;> simp [*, Set.ite, or_iff_right_of_imp (@h x)]
```

## `Set.pair_eq_pair_iff`
- file: `Mathlib/Data/Set/Basic.lean` (line 2073)
- proof_style: **subset_antisymm** | ext=False by_cases=False rw=False simp_set=False aesop=True
- likely_reusable_probe: `simp [subset_antisymm_iff, insert_subset_iff] <;> aesop`
- notes: official proof itself ends in aesop

```lean
theorem pair_eq_pair_iff {x y z w : α} :
    ({x, y} : Set α) = {z, w} ↔ x = z ∧ y = w ∨ x = w ∧ y = z := by
  simp [subset_antisymm_iff, insert_subset_iff]; aesop
```

## `Set.subset_insert_iff`
- file: `Mathlib/Data/Set/Basic.lean` (line 2051)
- proof_style: **rw_bridge** | ext=False by_cases=True rw=True simp_set=False aesop=False
- likely_reusable_probe: `rw-bridge (theorem-specific; see official proof)`
- notes: rewrite-bridge proof: depends on specific named lemmas, not closed by generic simp/aesop

```lean
theorem subset_insert_iff {s t : Set α} {x : α} :
    s ⊆ insert x t ↔ s ⊆ t ∨ (x ∈ s ∧ s \ {x} ⊆ t) := by
  rw [← diff_singleton_subset_iff]
  by_cases hx : x ∈ s
  · rw [and_iff_right hx, or_iff_right_of_imp diff_subset.trans]
  rw [diff_singleton_eq_self hx, or_iff_left_of_imp And.right]
```

## `Set.subset_singleton_iff_eq`
- file: `Mathlib/Data/Set/Basic.lean` (line 1446)
- proof_style: **simp_only** | ext=False by_cases=False rw=False simp_set=False aesop=False
- likely_reusable_probe: `aesop`

```lean
theorem subset_singleton_iff_eq {s : Set α} {x : α} : s ⊆ {x} ↔ s = ∅ ∨ s = {x} := by
  obtain rfl | hs := s.eq_empty_or_nonempty
  · exact ⟨fun _ => Or.inl rfl, fun _ => empty_subset _⟩
  · simp [eq_singleton_iff_nonempty_unique_mem, hs, hs.ne_empty]
```

## `Set.union_empty_iff`
- file: `Mathlib/Data/Set/Basic.lean` (line 853)
- proof_style: **simp_only** | ext=False by_cases=False rw=False simp_set=False aesop=False
- likely_reusable_probe: `aesop`

```lean
theorem union_empty_iff {s t : Set α} : s ∪ t = ∅ ↔ s = ∅ ∧ t = ∅ := by
  simp only [← subset_empty_iff]
  exact union_subset_iff
```

## `Set.antitoneOn_iff_antitone`
- file: `Mathlib/Data/Set/Basic.lean` (line 2368)
- proof_style: **simp_only** | ext=False by_cases=False rw=False simp_set=False aesop=False
- likely_reusable_probe: `aesop`

```lean
theorem antitoneOn_iff_antitone : AntitoneOn f s ↔
    Antitone fun a : s => f a := by
  simp [Antitone, AntitoneOn]
```

## `Set.ssubset_singleton_iff`
- file: `Mathlib/Data/Set/Basic.lean` (line 1456)
- proof_style: **rw_bridge** | ext=False by_cases=False rw=True simp_set=False aesop=False
- likely_reusable_probe: `rw-bridge (theorem-specific; see official proof)`
- notes: rewrite-bridge proof: depends on specific named lemmas, not closed by generic simp/aesop

```lean
theorem ssubset_singleton_iff {s : Set α} {x : α} : s ⊂ {x} ↔ s = ∅ := by
  rw [ssubset_iff_subset_ne, subset_singleton_iff_eq, or_and_right, and_not_self_iff, or_false_iff,
    and_iff_left_iff_imp]
  exact fun h => h ▸ (singleton_ne_empty _).symm
```

## `Set.ite_empty_right`
- file: `Mathlib/Data/Set/Basic.lean` (line 2307)
- proof_style: **simp_only** | ext=False by_cases=False rw=False simp_set=True aesop=False
- likely_reusable_probe: `simp [Set.ite]`
- notes: one-line `simp [Set.ite]`: RC1 simp does not unfold the irreducible Set.ite by default -> tactic gap

```lean
theorem ite_empty_right (t s : Set α) : t.ite s ∅ = s ∩ t := by simp [Set.ite]
```

## `Set.ite_inter`
- file: `Mathlib/Data/Set/Basic.lean` (line 2330)
- proof_style: **rw_bridge** | ext=False by_cases=False rw=True simp_set=False aesop=False
- likely_reusable_probe: `rw-bridge (theorem-specific; see official proof)`
- notes: rewrite-bridge proof: depends on specific named lemmas, not closed by generic simp/aesop

```lean
theorem ite_inter (t s₁ s₂ s : Set α) : t.ite (s₁ ∩ s) (s₂ ∩ s) = t.ite s₁ s₂ ∩ s := by
  rw [ite_inter_inter, ite_same]
```

## `Set.ite_inter_self`
- file: `Mathlib/Data/Set/Basic.lean` (line 2262)
- proof_style: **rw_bridge** | ext=False by_cases=False rw=True simp_set=True aesop=False
- likely_reusable_probe: `rw-bridge (theorem-specific; see official proof)`
- notes: rewrite-bridge proof: depends on specific named lemmas, not closed by generic simp/aesop

```lean
theorem ite_inter_self (t s s' : Set α) : t.ite s s' ∩ t = s ∩ t := by
  rw [Set.ite, union_inter_distrib_right, diff_inter_self, inter_assoc, inter_self, union_empty]
```

## `Set.ite_right`
- file: `Mathlib/Data/Set/Basic.lean` (line 2291)
- proof_style: **simp_only** | ext=False by_cases=False rw=False simp_set=True aesop=False
- likely_reusable_probe: `simp [Set.ite]`
- notes: one-line `simp [Set.ite]`: RC1 simp does not unfold the irreducible Set.ite by default -> tactic gap

```lean
theorem ite_right (s t : Set α) : s.ite t s = t ∩ s := by simp [Set.ite]
```
