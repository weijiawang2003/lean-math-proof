# SX3 Depth-2 Sequence Search — Live Results

- cases: `project/evolve/experiments/sx3/cases/sx3_negative_controls.json`
- families: SX3_SET_ITE_AESOP, SX3_SET_ITE_SIMPALL, SX3_SET_ITE_EXT, SX3_SET_ITE_EXT_AESOP, SX3_SET_EXT_AESOP, SX3_SET_EXT_SIMPALL, SX3_SET_IFF_CONSTRUCTOR_AESOP, SX3_SET_IFF_CONSTRUCTOR_SIMPALL, SX3_SET_SUBSET_ANTISYMM_AESOP, SX3_SET_SUBSET_ANTISYMM_SIMPALL
- theorems: 6 | live: 1 | result_hash: `754609f7e128`
- classification histogram: `{'unknown': 5, 'no_sequence_win': 1}`
- new_depth2_wins (0): (none)

| theorem | live | shape | class | best win |
|---|---|---|---|---|
| `Int.add_mul` | False | unknown | **unknown** | `` |
| `List.append_nil` | False | unknown | **unknown** | `` |
| `Multiset.cons_inj_left` | False | unknown | **unknown** | `` |
| `Multiset.toFinset_eq_singleton_iff` | True | unknown | **no_sequence_win** | `` |
| `Nat.add_comm` | False | unknown | **unknown** | `` |
| `Nat.mul_succ` | False | unknown | **unknown** | `` |

## `Int.add_mul`
- setup_error: AttributeError: 'NoneType' object has no attribute 'suffix'
^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 6, in __init__
  File "/opt/anaconda3/lib/python3.12/site-packages/lean_dojo/data_extraction/
- classification: **unknown** | best_win=`None`

## `List.append_nil`
- setup_error: AttributeError: 'NoneType' object has no attribute 'suffix'
^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 6, in __init__
  File "/opt/anaconda3/lib/python3.12/site-packages/lean_dojo/data_extraction/
- classification: **unknown** | best_win=`None`

## `Multiset.cons_inj_left`
- setup_error: AttributeError: 'NoneType' object has no attribute 'suffix'
^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 6, in __init__
  File "/opt/anaconda3/lib/python3.12/site-packages/lean_dojo/data_extraction/
- classification: **unknown** | best_win=`None`

## `Multiset.toFinset_eq_singleton_iff`
- initial goal: `α : Type u_1
β : Type u_2
γ : Type u_3
inst✝ : DecidableEq α
s✝ t s : Multiset α
a : α
⊢ s.toFinset = {a} ↔ card s ≠ 0 ∧ s = card s • {a}`
- classification: **no_sequence_win** | best_win=`None`
- controls:
    - `simp` -> proof_failed (solved=False)
    - `simp_all` -> proof_failed (solved=False)
    - `aesop` -> max_recursion (solved=False)
    - `classical <;> aesop` -> parse_error (solved=False)
    - `simp [Set.ite]` -> proof_failed (solved=False)

## `Nat.add_comm`
- setup_error: AttributeError: 'NoneType' object has no attribute 'suffix'
^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 6, in __init__
  File "/opt/anaconda3/lib/python3.12/site-packages/lean_dojo/data_extraction/
- classification: **unknown** | best_win=`None`

## `Nat.mul_succ`
- setup_error: AttributeError: 'NoneType' object has no attribute 'suffix'
^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 6, in __init__
  File "/opt/anaconda3/lib/python3.12/site-packages/lean_dojo/data_extraction/
- classification: **unknown** | best_win=`None`

> Live LeanDojo depth-2 sequences. No solve is a confirmed win; minimal-sufficient relabel + RC2-baseline comparison required before any promotion. RC1/RC2 production configs untouched.