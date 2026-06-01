# SX3 Depth-2 Sequence Search — Live Results

- cases: `project/evolve/experiments/sx3/cases/sx3_deferred_set_ite_cases.json`
- families: SX3_SET_ITE_AESOP, SX3_SET_ITE_SIMPALL, SX3_SET_ITE_EXT
- theorems: 4 | live: 4 | result_hash: `c0144cd63fd5`
- classification histogram: `{'new_depth2_win': 4}`
- new_depth2_wins (4): `Set.ite_compl`, `Set.ite_inter`, `Set.ite_inter_compl_self`, `Set.ite_inter_self`

| theorem | live | shape | class | best win |
|---|---|---|---|---|
| `Set.ite_compl` | True | set_equality | **new_depth2_win** | `simp [Set.ite] <;> aesop` |
| `Set.ite_inter` | True | set_equality | **new_depth2_win** | `simp [Set.ite] <;> aesop` |
| `Set.ite_inter_compl_self` | True | set_equality | **new_depth2_win** | `simp [Set.ite] <;> aesop` |
| `Set.ite_inter_self` | True | set_equality | **new_depth2_win** | `simp [Set.ite] <;> aesop` |

## `Set.ite_compl`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s s' : Set α
⊢ tᶜ.ite s s' = t.ite s' s`
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

## `Set.ite_inter`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ s : Set α
⊢ t.ite (s₁ ∩ s) (s₂ ∩ s) = t.ite s₁ s₂ ∩ s`
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

## `Set.ite_inter_compl_self`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s s' : Set α
⊢ t.ite s s' ∩ tᶜ = s' ∩ tᶜ`
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

## `Set.ite_inter_self`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s s' : Set α
⊢ t.ite s s' ∩ t = s ∩ t`
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

> Live LeanDojo depth-2 sequences. No solve is a confirmed win; minimal-sufficient relabel + RC2-baseline comparison required before any promotion. RC1/RC2 production configs untouched.