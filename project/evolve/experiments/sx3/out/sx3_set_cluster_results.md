# SX3 Depth-2 Sequence Search — Live Results

- cases: `project/evolve/experiments/sx3/cases/sx3_set_failure_cluster_cases.json`
- families: SX3_SET_EXT_AESOP, SX3_SET_EXT_SIMPALL, SX3_SET_IFF_CONSTRUCTOR_AESOP, SX3_SET_IFF_CONSTRUCTOR_SIMPALL, SX3_SET_SUBSET_ANTISYMM_AESOP, SX3_SET_SUBSET_ANTISYMM_SIMPALL
- theorems: 12 | live: 11 | result_hash: `071a3497ab3c`
- classification histogram: `{'unknown': 1, 'no_sequence_win': 8, 'baseline_duplicate': 3}`
- new_depth2_wins (0): (none)

| theorem | live | shape | class | best win |
|---|---|---|---|---|
| `Set.antitoneOn_iff_antitone` | True | set_iff | **no_sequence_win** | `` |
| `Set.diff_singleton_subset_iff` | False |  | **unknown** | `` |
| `Set.diff_union_inter` | True | set_equality | **no_sequence_win** | `` |
| `Set.insert_diff_eq_singleton` | True | set_equality | **baseline_duplicate** | `ext x <;> aesop` |
| `Set.insert_diff_of_mem` | True | set_equality | **baseline_duplicate** | `ext x <;> aesop` |
| `Set.pair_diff_left` | True | set_equality | **baseline_duplicate** | `ext x <;> aesop` |
| `Set.pair_eq_pair_iff` | True | set_iff | **no_sequence_win** | `` |
| `Set.powerset_singleton` | True | set_equality | **no_sequence_win** | `` |
| `Set.ssubset_singleton_iff` | True | set_iff | **no_sequence_win** | `` |
| `Set.subset_insert_iff` | True | set_iff | **no_sequence_win** | `` |
| `Set.subset_singleton_iff_eq` | True | set_iff | **no_sequence_win** | `` |
| `Set.union_empty_iff` | True | set_iff | **no_sequence_win** | `` |

## `Set.antitoneOn_iff_antitone`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t t₁ t₂ u : Set α
inst✝¹ : Preorder α
inst✝ : Preorder β
f : α → β
⊢ AntitoneOn f s ↔ Antitone fun a => f ↑a`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_IFF_CONSTRUCTOR_AESOP] `constructor <;> intro h <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_IFF_CONSTRUCTOR_SIMPALL] `constructor <;> intro h <;> simp_all` -> proof_failed (solved=False)

## `Set.diff_singleton_subset_iff`
- setup_error: no worker output (rc=1); OS-killed at 275s
- classification: **unknown** | best_win=`None`

## `Set.diff_union_inter`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
⊢ s \ t ∪ s ∩ t = s`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_EXT_AESOP] `ext x <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_EXT_SIMPALL] `ext x <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_SUBSET_ANTISYMM_AESOP] `apply Set.Subset.antisymm <;> intro x <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_SUBSET_ANTISYMM_SIMPALL] `apply Set.Subset.antisymm <;> intro x <;> simp_all` -> proof_failed (solved=False)

## `Set.insert_diff_eq_singleton`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a✝ b : α
s✝ s₁ s₂ t t₁ t₂ u : Set α
a : α
s : Set α
h : a ∉ s
⊢ insert a s \ s = {a}`
- classification: **baseline_duplicate** | best_win=`ext x <;> aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> solved (solved=True)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_EXT_AESOP] `ext x <;> aesop` -> solved (solved=True)
    - [SX3_SET_EXT_SIMPALL] `ext x <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_SUBSET_ANTISYMM_AESOP] `apply Set.Subset.antisymm <;> intro x <;> aesop` -> solved (solved=True)
    - [SX3_SET_SUBSET_ANTISYMM_SIMPALL] `apply Set.Subset.antisymm <;> intro x <;> simp_all` -> proof_failed (solved=False)

## `Set.insert_diff_of_mem`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s : Set α
h : a ∈ t
⊢ insert a s \ t = s \ t`
- classification: **baseline_duplicate** | best_win=`ext x <;> aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> solved (solved=True)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_EXT_AESOP] `ext x <;> aesop` -> solved (solved=True)
    - [SX3_SET_EXT_SIMPALL] `ext x <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_SUBSET_ANTISYMM_AESOP] `apply Set.Subset.antisymm <;> intro x <;> aesop` -> solved (solved=True)
    - [SX3_SET_SUBSET_ANTISYMM_SIMPALL] `apply Set.Subset.antisymm <;> intro x <;> simp_all` -> proof_failed (solved=False)

## `Set.pair_diff_left`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t t₁ t₂ u : Set α
hne : a ≠ b
⊢ {a, b} \ {a} = {b}`
- classification: **baseline_duplicate** | best_win=`ext x <;> aesop`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> solved (solved=True)
    - `aesop` -> solved (solved=True)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> unknown_ident (solved=False)
- gated sequences:
    - [SX3_SET_EXT_AESOP] `ext x <;> aesop` -> solved (solved=True)
    - [SX3_SET_EXT_SIMPALL] `ext x <;> simp_all` -> solved (solved=True)
    - [SX3_SET_SUBSET_ANTISYMM_AESOP] `apply Set.Subset.antisymm <;> intro x <;> aesop` -> solved (solved=True)
    - [SX3_SET_SUBSET_ANTISYMM_SIMPALL] `apply Set.Subset.antisymm <;> intro x <;> simp_all` -> solved (solved=True)

## `Set.pair_eq_pair_iff`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t t₁ t₂ u : Set α
x y z w : α
⊢ {x, y} = {z, w} ↔ x = z ∧ y = w ∨ x = w ∧ y = z`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_IFF_CONSTRUCTOR_AESOP] `constructor <;> intro h <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_IFF_CONSTRUCTOR_SIMPALL] `constructor <;> intro h <;> simp_all` -> proof_failed (solved=False)

## `Set.powerset_singleton`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t t₁ t₂ u : Set α
x : α
⊢ 𝒫{x} = {∅, {x}}`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_EXT_AESOP] `ext x <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_EXT_SIMPALL] `ext x <;> simp_all` -> proof_failed (solved=False)
    - [SX3_SET_SUBSET_ANTISYMM_AESOP] `apply Set.Subset.antisymm <;> intro x <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_SUBSET_ANTISYMM_SIMPALL] `apply Set.Subset.antisymm <;> intro x <;> simp_all` -> proof_failed (solved=False)

## `Set.ssubset_singleton_iff`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s : Set α
x : α
⊢ s ⊂ {x} ↔ s = ∅`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_IFF_CONSTRUCTOR_AESOP] `constructor <;> intro h <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_IFF_CONSTRUCTOR_SIMPALL] `constructor <;> intro h <;> simp_all` -> proof_failed (solved=False)

## `Set.subset_insert_iff`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
x : α
⊢ s ⊆ insert x t ↔ s ⊆ t ∨ x ∈ s ∧ s \ {x} ⊆ t`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> unknown_ident (solved=False)
- gated sequences:
    - [SX3_SET_IFF_CONSTRUCTOR_AESOP] `constructor <;> intro h <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_IFF_CONSTRUCTOR_SIMPALL] `constructor <;> intro h <;> simp_all` -> proof_failed (solved=False)

## `Set.subset_singleton_iff_eq`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s : Set α
x : α
⊢ s ⊆ {x} ↔ s = ∅ ∨ s = {x}`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> unknown_ident (solved=False)
- gated sequences:
    - [SX3_SET_IFF_CONSTRUCTOR_AESOP] `constructor <;> intro h <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_IFF_CONSTRUCTOR_SIMPALL] `constructor <;> intro h <;> simp_all` -> proof_failed (solved=False)

## `Set.union_empty_iff`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
⊢ s ∪ t = ∅ ↔ s = ∅ ∧ t = ∅`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> proof_failed (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)
- gated sequences:
    - [SX3_SET_IFF_CONSTRUCTOR_AESOP] `constructor <;> intro h <;> aesop` -> proof_failed (solved=False)
    - [SX3_SET_IFF_CONSTRUCTOR_SIMPALL] `constructor <;> intro h <;> simp_all` -> proof_failed (solved=False)

> Live LeanDojo depth-2 sequences. No solve is a confirmed win; minimal-sufficient relabel + RC2-baseline comparison required before any promotion. RC1/RC2 production configs untouched.