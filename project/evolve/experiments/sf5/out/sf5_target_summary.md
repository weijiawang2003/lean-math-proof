# SF5 — missing-bridge target set

- targets: **20**
- confirmed literal-RC2 failures (pool): 27
- TR2 MISSING_BRIDGE_LEMMA_CANDIDATE: 20
- SF4 POSSIBLE_MISSING_BRIDGE_LEMMA cluster members: 19

## Clusters

### `Set__iff__iff` (size 16)
- Set.Nonempty.subset_pair_iff_eq
- Set.antitoneOn_iff_antitone
- Set.diff_singleton_subset_iff
- Set.monotoneOn_iff_monotone
- Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le
- Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt
- Set.pair_eq_pair_iff
- Set.ssubset_iff_insert
- Set.ssubset_iff_sdiff_singleton
- Set.ssubset_singleton_iff
- Set.strictAntiOn_iff_strictAnti
- Set.strictMonoOn_iff_strictMono
- Set.subset_insert_iff
- Set.subset_pair_iff_eq
- Set.subset_singleton_iff_eq
- Set.union_empty_iff

### `Set__ite_if__subset` (size 3)
- Set.ite_eq_of_subset_left
- Set.ite_eq_of_subset_right
- Set.subset_ite

### `Multiset__iff__iff` (size 1)
- Multiset.toFinset_eq_singleton_iff

## Targets

| full_name | namespace | cluster | iff | subset | mono | goal |
|---|---|---|---|---|---|---|
| Multiset.toFinset_eq_singleton_iff | Multiset | Multiset__iff__iff | Y |  |  | theorem toFinset_eq_singleton_iff (s : Multiset α) (a : α) : |
| Set.Nonempty.subset_pair_iff_eq | Set | Set__iff__iff | Y | Y |  | theorem subset_pair_iff_eq {x y : α} : s ⊆ {x, y} ↔ s = ∅ ∨  |
| Set.antitoneOn_iff_antitone | Set | Set__iff__iff | Y |  |  | theorem antitoneOn_iff_antitone : AntitoneOn f s ↔ Antitone  |
| Set.diff_singleton_subset_iff | Set | Set__iff__iff | Y | Y |  | theorem diff_singleton_subset_iff {x : α} {s t : Set α} : s  |
| Set.ite_eq_of_subset_left | Set | Set__ite_if__subset |  |  |  | ⊢ t.ite s₁ s₂ = s₁ ∪ s₂ \ t |
| Set.ite_eq_of_subset_right | Set | Set__ite_if__subset |  |  |  | ⊢ t.ite s₁ s₂ = s₁ ∩ t ∪ s₂ |
| Set.monotoneOn_iff_monotone | Set | Set__iff__iff | Y |  | Y | theorem monotoneOn_iff_monotone : MonotoneOn f s ↔ Monotone  |
| Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le | Set | Set__iff__iff | Y |  | Y | theorem not_monotoneOn_not_antitoneOn_iff_exists_le_le : ¬Mo |
| Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt | Set | Set__iff__iff | Y |  | Y | theorem not_monotoneOn_not_antitoneOn_iff_exists_lt_lt : ¬Mo |
| Set.pair_eq_pair_iff | Set | Set__iff__iff | Y |  |  | theorem pair_eq_pair_iff {x y z w : α} : ({x, y} : Set α) =  |
| Set.ssubset_iff_insert | Set | Set__iff__iff | Y | Y |  | theorem ssubset_iff_insert {s t : Set α} : s ⊂ t ↔ ∃ a ∉ s,  |
| Set.ssubset_iff_sdiff_singleton | Set | Set__iff__iff | Y | Y |  | lemma ssubset_iff_sdiff_singleton : s ⊂ t ↔ ∃ a ∈ t, s ⊆ t \ |
| Set.ssubset_singleton_iff | Set | Set__iff__iff | Y | Y |  | theorem ssubset_singleton_iff {s : Set α} {x : α} : s ⊂ {x}  |
| Set.strictAntiOn_iff_strictAnti | Set | Set__iff__iff | Y |  |  | theorem strictAntiOn_iff_strictAnti : StrictAntiOn f s ↔ Str |
| Set.strictMonoOn_iff_strictMono | Set | Set__iff__iff | Y |  |  | theorem strictMonoOn_iff_strictMono : StrictMonoOn f s ↔ Str |
| Set.subset_insert_iff | Set | Set__iff__iff | Y | Y |  | theorem subset_insert_iff {s t : Set α} {x : α} : s ⊆ insert |
| Set.subset_ite | Set | Set__ite_if__subset | Y | Y |  | theorem subset_ite {t s s' u : Set α} : u ⊆ t.ite s s' ↔ u ∩ |
| Set.subset_pair_iff_eq | Set | Set__iff__iff | Y | Y |  | theorem subset_pair_iff_eq {x y : α} : s ⊆ {x, y} ↔ s = ∅ ∨  |
| Set.subset_singleton_iff_eq | Set | Set__iff__iff | Y | Y |  | theorem subset_singleton_iff_eq {s : Set α} {x : α} : s ⊆ {x |
| Set.union_empty_iff | Set | Set__iff__iff | Y |  |  | theorem union_empty_iff {s t : Set α} : s ∪ t = ∅ ↔ s = ∅ ∧  |
