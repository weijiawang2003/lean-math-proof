# SX2 — SET2 Live Eval Results

- cases: `project/evolve/experiments/sf2/out/set_cluster_deep_dive/selected_cases.json`
- production default emits: **False** (SET2 off-by-default; forced on for this experiment)
- total=12 live=12 | RC1-proxy solved=0 | SET2 solved=2 | **SET2 new wins over RC1=2** | regressions=0 | off-gate=0
- gate precision: {'emitted_and_solved': 2, 'emitted_and_failed': 10, 'not_emitted': 0}
- rc1_solved == baseline battery closes goal; valid RC1 proxy for Set/Basic surfaces (WX3/MX2 do not apply). The 12 SF2 selected cases are known RC1-failed.

| theorem | shape | rc1 | emitted | gate | set2_tactic | set2_solved | off_gate |
|---|---|---|---|---|---|---|---|
| `Set.diff_singleton_subset_iff` | equality | False | True | None | `` | False | False |
| `Set.ite_eq_of_subset_left` | equality | False | True | None | `` | False | False |
| `Set.pair_eq_pair_iff` | equality | False | True | None | `` | False | False |
| `Set.subset_insert_iff` | equality | False | True | None | `` | False | False |
| `Set.subset_singleton_iff_eq` | equality | False | True | None | `` | False | False |
| `Set.union_empty_iff` | equality | False | True | None | `` | False | False |
| `Set.antitoneOn_iff_antitone` | iff | False | True | None | `` | False | False |
| `Set.ssubset_singleton_iff` | iff | False | True | None | `` | False | False |
| `Set.ite_empty_right` | membership | False | True | SET_ITE_SIMP | `simp [Set.ite]` | True | False |
| `Set.ite_inter` | membership | False | True | None | `` | False | False |
| `Set.ite_inter_self` | membership | False | True | None | `` | False | False |
| `Set.ite_right` | membership | False | True | SET_ITE_SIMP | `simp [Set.ite]` | True | False |

## `Set.diff_singleton_subset_iff`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u : Set α
x : α
s t : Set α
⊢ s \ {x} ⊆ t ↔ s ⊆ insert x t`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.ite_eq_of_subset_left`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ : Set α
h : s₁ ⊆ s₂
⊢ t.ite s₁ s₂ = s₁ ∪ s₂ \ t`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_ITE_SIMP:proof_failed', 'SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.pair_eq_pair_iff`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t t₁ t₂ u : Set α
x y z w : α
⊢ {x, y} = {z, w} ↔ x = z ∧ y = w ∨ x = w ∧ y = z`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.subset_insert_iff`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
x : α
⊢ s ⊆ insert x t ↔ s ⊆ t ∨ x ∈ s ∧ s \ {x} ⊆ t`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.subset_singleton_iff_eq`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s : Set α
x : α
⊢ s ⊆ {x} ↔ s = ∅ ∨ s = {x}`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.union_empty_iff`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
⊢ s ∪ t = ∅ ↔ s = ∅ ∧ t = ∅`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.antitoneOn_iff_antitone`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t t₁ t₂ u : Set α
inst✝¹ : Preorder α
inst✝ : Preorder β
f : α → β
⊢ AntitoneOn f s ↔ Antitone fun a => f ↑a`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.ssubset_singleton_iff`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s : Set α
x : α
⊢ s ⊂ {x} ↔ s = ∅`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.ite_empty_right`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s : Set α
⊢ t.ite s ∅ = s ∩ t`
- rc1(proxy)=False (by `None`) | set2_solved=True via gate SET_ITE_SIMP
- emitted gates → results: ['SET_ITE_SIMP:solved', 'SET_EXT_SIMP:-', 'SET_SUBSET_ANTISYMM:-']

## `Set.ite_inter`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ s : Set α
⊢ t.ite (s₁ ∩ s) (s₂ ∩ s) = t.ite s₁ s₂ ∩ s`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_ITE_SIMP:proof_failed', 'SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.ite_inter_self`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s s' : Set α
⊢ t.ite s s' ∩ t = s ∩ t`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_ITE_SIMP:proof_failed', 'SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.ite_right`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
⊢ s.ite t s = t ∩ s`
- rc1(proxy)=False (by `None`) | set2_solved=True via gate SET_ITE_SIMP
- emitted gates → results: ['SET_ITE_SIMP:solved', 'SET_EXT_SIMP:-', 'SET_SUBSET_ANTISYMM:-']

> No solve is promotion-confirmed; NS23 minimal relabel required. RC1/NS24/NS9 untouched. SET2 off-by-default.