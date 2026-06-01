# SX3 Depth-2 Sequence Search — Live Results

- cases: `project/evolve/experiments/sx3/cases/sx3_set_ite_fresh_holdout.json`
- families: SX3_SET_ITE_AESOP, SX3_SET_ITE_SIMPALL, SX3_SET_ITE_EXT, SX3_SET_ITE_EXT_AESOP
- theorems: 13 | live: 13 | result_hash: `ed6b9ef789a0`
- classification histogram: `{'no_sequence_win': 4, 'new_depth2_win': 1, 'single_step_duplicate': 1, 'baseline_duplicate': 7}`
- new_depth2_wins (1): `Set.ite_inter_inter`

| theorem | live | shape | class | best win |
|---|---|---|---|---|
| `Set.ite_eq_of_subset_left` | True | set_equality | **no_sequence_win** | `` |
| `Set.ite_eq_of_subset_right` | True | set_equality | **no_sequence_win** | `` |
| `Set.ite_inter_inter` | True | set_equality | **new_depth2_win** | `simp [Set.ite] <;> aesop` |
| `Set.ite_inter_of_inter_eq` | True | set_equality | **no_sequence_win** | `` |
| `Set.ite_univ` | True | set_equality | **single_step_duplicate** | `simp [Set.ite] <;> aesop` |
| `Set.mem_dite` | True | set_membership_iff | **baseline_duplicate** | `aesop` |
| `Set.mem_dite_empty_left` | True | set_membership_iff | **baseline_duplicate** | `aesop` |
| `Set.mem_dite_empty_right` | True | set_membership_iff | **baseline_duplicate** | `aesop` |
| `Set.mem_dite_univ_left` | True | set_membership_iff | **baseline_duplicate** | `aesop` |
| `Set.mem_dite_univ_right` | True | set_membership_iff | **baseline_duplicate** | `aesop` |
| `Set.mem_ite_empty_left` | True | set_membership_iff | **baseline_duplicate** | `aesop` |
| `Set.mem_ite_empty_right` | True | set_membership_iff | **baseline_duplicate** | `aesop` |
| `Set.subset_ite` | True | set_subset_iff | **no_sequence_win** | `` |

## `Set.ite_eq_of_subset_left`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ : Set α
h : s₁ ⊆ s₂
⊢ t.ite s₁ s₂ = s₁ ∪ s₂ \ t`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> proof_failed (solved=False)

## `Set.ite_eq_of_subset_right`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ : Set α
h : s₂ ⊆ s₁
⊢ t.ite s₁ s₂ = s₁ ∩ t ∪ s₂`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> proof_failed (solved=False)

## `Set.ite_inter_inter`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ s₁' s₂' : Set α
⊢ t.ite (s₁ ∩ s₂) (s₁' ∩ s₂') = t.ite s₁ s₁' ∩ t.ite s₂ s₂'`
- classification: **new_depth2_win** | best_win=`simp [Set.ite] <;> aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> solved (solved=True)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> solved (solved=True)

## `Set.ite_inter_of_inter_eq`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ s : Set α
h : s₁ ∩ s = s₂ ∩ s
⊢ t.ite s₁ s₂ ∩ s = s₁ ∩ s`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> proof_failed (solved=False)

## `Set.ite_univ`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s s' : Set α
⊢ univ.ite s s' = s`
- classification: **single_step_duplicate** | best_win=`simp [Set.ite] <;> aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> solved (solved=True)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> solved (solved=True)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> solved (solved=True)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> solved (solved=True)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> solved (solved=True)

## `Set.mem_dite`
- initial goal: `α : Type u
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
⊢ (x ∈ if h : p then s h else t h) ↔ (∀ (h : p), x ∈ s h`
- classification: **baseline_duplicate** | best_win=`aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> solved (solved=True)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> ext_not_applicable (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> ext_not_applicable (solved=False)

## `Set.mem_dite_empty_left`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t✝ t₁ t₂ u : Set α
p : Prop
inst✝ : Decidable p
t : ¬p → Set α
x : α
⊢ (x ∈ if h : p then ∅ else t h) ↔ ∃ (h : ¬p), x ∈ t h`
- classification: **baseline_duplicate** | best_win=`aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> solved (solved=True)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> ext_not_applicable (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> ext_not_applicable (solved=False)

## `Set.mem_dite_empty_right`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t✝ t₁ t₂ u : Set α
p : Prop
inst✝ : Decidable p
t : p → Set α
x : α
⊢ (x ∈ if h : p then t h else ∅) ↔ ∃ (h : p), x ∈ t h`
- classification: **baseline_duplicate** | best_win=`aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> solved (solved=True)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> ext_not_applicable (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> ext_not_applicable (solved=False)

## `Set.mem_dite_univ_left`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t✝ t₁ t₂ u : Set α
p : Prop
inst✝ : Decidable p
t : ¬p → Set α
x : α
⊢ (x ∈ if h : p then univ else t h) ↔ ∀ (h : ¬p), x ∈ t h`
- classification: **baseline_duplicate** | best_win=`aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> solved (solved=True)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> ext_not_applicable (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> ext_not_applicable (solved=False)

## `Set.mem_dite_univ_right`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t✝ t₁ t₂ u : Set α
p : Prop
inst✝ : Decidable p
t : p → Set α
x : α
⊢ (x ∈ if h : p then t h else univ) ↔ ∀ (h : p), x ∈ t h`
- classification: **baseline_duplicate** | best_win=`aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> solved (solved=True)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> ext_not_applicable (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> ext_not_applicable (solved=False)

## `Set.mem_ite_empty_left`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t✝ t₁ t₂ u : Set α
p : Prop
inst✝ : Decidable p
t : Set α
x : α
⊢ (x ∈ if p then ∅ else t) ↔ ¬p ∧ x ∈ t`
- classification: **baseline_duplicate** | best_win=`aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> solved (solved=True)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> ext_not_applicable (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> ext_not_applicable (solved=False)

## `Set.mem_ite_empty_right`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t✝ t₁ t₂ u : Set α
p : Prop
inst✝ : Decidable p
t : Set α
x : α
⊢ (x ∈ if p then t else ∅) ↔ p ∧ x ∈ t`
- classification: **baseline_duplicate** | best_win=`aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> solved (solved=True)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> ext_not_applicable (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> ext_not_applicable (solved=False)

## `Set.subset_ite`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u✝ t s s' u : Set α
⊢ u ⊆ t.ite s s' ↔ u ∩ t ⊆ s ∧ u \ t ⊆ s'`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_ITE_AESOP] `simp [Set.ite] <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_ITE_SIMPALL] `simp [Set.ite] <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_ITE_EXT] `ext x <;> simp [Set.ite]` -> ext_not_applicable (solved=False)
    - [SX3_SET_ITE_EXT_AESOP] `ext x <;> simp [Set.ite] <;> aesop` -> ext_not_applicable (solved=False)

> Live LeanDojo depth-2 sequences. No solve is a confirmed win; minimal-sufficient relabel + RC2-baseline comparison required before any promotion. RC1/RC2 production configs untouched.