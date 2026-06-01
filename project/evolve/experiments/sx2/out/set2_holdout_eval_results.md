# SX2 — SET2 Live Eval Results

- cases: `project/evolve/experiments/sx2/out/set2_holdout_cases.json`
- production default emits: **False** (SET2 off-by-default; forced on for this experiment)
- total=20 live=20 | RC1-proxy solved=9 | SET2 solved=4 | **SET2 new wins over RC1=4** | regressions=0 | off-gate=0
- gate precision: {'emitted_and_solved': 4, 'emitted_and_failed': 16, 'not_emitted': 0}
- rc1_solved == baseline battery closes goal; valid RC1 proxy for Set/Basic surfaces (WX3/MX2 do not apply). The 12 SF2 selected cases are known RC1-failed.

| theorem | shape | rc1 | emitted | gate | set2_tactic | set2_solved | off_gate |
|---|---|---|---|---|---|---|---|
| `Set.ite_compl` | unknown_pre_live | False | True | None | `` | False | False |
| `Set.ite_empty` | unknown_pre_live | False | True | SET_ITE_SIMP | `simp [Set.ite]` | True | False |
| `Set.ite_empty_left` | unknown_pre_live | False | True | SET_ITE_SIMP | `simp [Set.ite]` | True | False |
| `Set.ite_inter_of_inter_eq` | unknown_pre_live | False | True | None | `` | False | False |
| `Set.ite_left` | unknown_pre_live | False | True | SET_ITE_SIMP | `simp [Set.ite]` | True | False |
| `Set.mem_dite` | unknown_pre_live | True | True | None | `` | False | False |
| `Set.mem_dite_empty_left` | unknown_pre_live | True | True | None | `` | False | False |
| `Set.mem_dite_empty_right` | unknown_pre_live | True | True | None | `` | False | False |
| `Set.Nonempty.subset_pair_iff_eq` | unknown_pre_live | False | True | None | `` | False | False |
| `Set.diff_union_inter` | unknown_pre_live | False | True | None | `` | False | False |
| `Set.eq_of_inclusion_surjective` | unknown_pre_live | False | True | None | `` | False | False |
| `Set.inclusion_inclusion` | unknown_pre_live | True | True | None | `` | False | False |
| `Set.inclusion_right` | unknown_pre_live | False | True | SET_EXT_SIMP | `ext x <;> simp` | True | False |
| `Set.inclusion_self` | unknown_pre_live | True | True | None | `` | False | False |
| `Set.insert_diff_eq_singleton` | unknown_pre_live | True | True | None | `` | False | False |
| `Set.insert_diff_of_mem` | unknown_pre_live | True | True | None | `` | False | False |
| `Set.insert_diff_of_not_mem` | unknown_pre_live | True | True | None | `` | False | False |
| `Set.insert_diff_self_of_not_mem` | unknown_pre_live | True | True | None | `` | False | False |
| `Set.monotoneOn_iff_monotone` | unknown_pre_live | False | True | None | `` | False | False |
| `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt` | unknown_pre_live | False | True | None | `` | False | False |

## `Set.ite_compl`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s s' : Set α
⊢ tᶜ.ite s s' = t.ite s' s`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_ITE_SIMP:proof_failed', 'SET_EXT_SIMP:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.ite_empty`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s s' : Set α
⊢ ∅.ite s s' = s'`
- rc1(proxy)=False (by `None`) | set2_solved=True via gate SET_ITE_SIMP
- emitted gates → results: ['SET_ITE_SIMP:solved', 'SET_EXT_SIMP:-']

## `Set.ite_empty_left`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s : Set α
⊢ t.ite ∅ s = s \ t`
- rc1(proxy)=False (by `None`) | set2_solved=True via gate SET_ITE_SIMP
- emitted gates → results: ['SET_ITE_SIMP:solved', 'SET_EXT_SIMP:-', 'SET_SUBSET_ANTISYMM:-']

## `Set.ite_inter_of_inter_eq`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ s : Set α
h : s₁ ∩ s = s₂ ∩ s
⊢ t.ite s₁ s₂ ∩ s = s₁ ∩ s`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_ITE_SIMP:proof_failed', 'SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.ite_left`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
⊢ s.ite s t = s ∪ t`
- rc1(proxy)=False (by `None`) | set2_solved=True via gate SET_ITE_SIMP
- emitted gates → results: ['SET_ITE_SIMP:solved', 'SET_EXT_SIMP:-', 'SET_SUBSET_ANTISYMM:-']

## `Set.mem_dite`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u : Set α
p : Prop
inst✝ : Decidable p
s : p → Set α
t : ¬p → Set α
x : α
⊢ (x ∈ if h : p then s h else t h) ↔`
- rc1(proxy)=True (by `aesop`) | set2_solved=False via gate None
- emitted gates → results: ['SET_ITE_SIMP:proof_failed', 'SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.mem_dite_empty_left`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t✝ t₁ t₂ u : Set α
p : Prop
inst✝ : Decidable p
t : ¬p → Set α
x : α
⊢ (x ∈ if h : p then ∅ else t h) ↔ ∃ (h : ¬p), x ∈ `
- rc1(proxy)=True (by `aesop`) | set2_solved=False via gate None
- emitted gates → results: ['SET_ITE_SIMP:proof_failed', 'SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.mem_dite_empty_right`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t✝ t₁ t₂ u : Set α
p : Prop
inst✝ : Decidable p
t : p → Set α
x : α
⊢ (x ∈ if h : p then t h else ∅) ↔ ∃ (h : p), x ∈ t `
- rc1(proxy)=True (by `aesop`) | set2_solved=False via gate None
- emitted gates → results: ['SET_ITE_SIMP:proof_failed', 'SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.Nonempty.subset_pair_iff_eq`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t t₁ t₂ u : Set α
hs : s.Nonempty
⊢ s ⊆ {a, b} ↔ s = {a} ∨ s = {b} ∨ s = {a, b}`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.diff_union_inter`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
⊢ s \ t ∪ s ∩ t = s`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.eq_of_inclusion_surjective`
- goal: `α : Type u_1
s✝ t✝ u s t : Set α
h : s ⊆ t
h_surj : Surjective (inclusion h)
⊢ s = t`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.inclusion_inclusion`
- goal: `α : Type u_1
s t u : Set α
hst : s ⊆ t
htu : t ⊆ u
x : ↑s
⊢ inclusion htu (inclusion hst x) = inclusion ⋯ x`
- rc1(proxy)=True (by `aesop`) | set2_solved=False via gate None
- emitted gates → results: ['SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.inclusion_right`
- goal: `α : Type u_1
s t u : Set α
h : s ⊆ t
x : ↑t
m : ↑x ∈ s
⊢ inclusion h ⟨↑x, m⟩ = x`
- rc1(proxy)=False (by `None`) | set2_solved=True via gate SET_EXT_SIMP
- emitted gates → results: ['SET_EXT_SIMP:solved', 'SET_SUBSET_ANTISYMM:-']

## `Set.inclusion_self`
- goal: `α : Type u_1
s t u : Set α
x : ↑s
⊢ inclusion ⋯ x = x`
- rc1(proxy)=True (by `aesop`) | set2_solved=False via gate None
- emitted gates → results: ['SET_EXT_SIMP:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.insert_diff_eq_singleton`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a✝ b : α
s✝ s₁ s₂ t t₁ t₂ u : Set α
a : α
s : Set α
h : a ∉ s
⊢ insert a s \ s = {a}`
- rc1(proxy)=True (by `aesop`) | set2_solved=False via gate None
- emitted gates → results: ['SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.insert_diff_of_mem`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s : Set α
h : a ∈ t
⊢ insert a s \ t = s \ t`
- rc1(proxy)=True (by `aesop`) | set2_solved=False via gate None
- emitted gates → results: ['SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.insert_diff_of_not_mem`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s : Set α
h : a ∉ t
⊢ insert a s \ t = insert a (s \ t)`
- rc1(proxy)=True (by `aesop`) | set2_solved=False via gate None
- emitted gates → results: ['SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.insert_diff_self_of_not_mem`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a✝ b : α
s✝ s₁ s₂ t t₁ t₂ u : Set α
a : α
s : Set α
h : a ∉ s
⊢ insert a s \ {a} = s`
- rc1(proxy)=True (by `simp_all`) | set2_solved=False via gate None
- emitted gates → results: ['SET_EXT_SIMP:proof_failed', 'SET_SUBSET_ANTISYMM:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.monotoneOn_iff_monotone`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t t₁ t₂ u : Set α
inst✝¹ : Preorder α
inst✝ : Preorder β
f : α → β
⊢ MonotoneOn f s ↔ Monotone fun a => f ↑a`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

## `Set.not_monotoneOn_not_antitoneOn_iff_exists_lt_lt`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t t₁ t₂ u : Set α
inst✝¹ : LinearOrder α
inst✝ : LinearOrder β
f : α → β
⊢ ¬MonotoneOn f s ∧ ¬AntitoneOn f s ↔
    ∃ a ∈`
- rc1(proxy)=False (by `None`) | set2_solved=False via gate None
- emitted gates → results: ['SET_IFF_CONSTRUCTOR:proof_failed']
- notes: ['SET2 emitted but did not close the goal']

> No solve is promotion-confirmed; NS23 minimal relabel required. RC1/NS24/NS9 untouched. SET2 off-by-default.