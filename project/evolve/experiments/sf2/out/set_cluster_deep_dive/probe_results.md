# SF2 Set Cluster — Live Probe Ladder Results

- theorems: 12 | live: 12 | solved: 10
- gap histogram: `{'search_depth_gap': 5, 'tactic_gap': 5, 'needs_deeper_search': 2}`

| theorem | shape | live | solved | gap | winning probe | min-sufficient |
|---|---|---|---|---|---|---|
| `Set.diff_singleton_subset_iff` | equality | True | True | search_depth_gap | `rw [← union_singleton, union_comm] <;> a` | `rw [← union_singleton, union_c` |
| `Set.ite_eq_of_subset_left` | equality | True | True | search_depth_gap | `ext x <;> by_cases hx : x ∈ t <;> simp [` | `ext x <;> by_cases hx : x ∈ t ` |
| `Set.pair_eq_pair_iff` | equality | True | True | tactic_gap | `simp [subset_antisymm_iff, insert_subset` | `simp [subset_antisymm_iff, ins` |
| `Set.subset_insert_iff` | equality | True | False | needs_deeper_search | `` | `` |
| `Set.subset_singleton_iff_eq` | equality | True | False | needs_deeper_search | `` | `` |
| `Set.union_empty_iff` | equality | True | True | tactic_gap | `simp only [← subset_empty_iff, union_sub` | `simp only [← subset_empty_iff,` |
| `Set.antitoneOn_iff_antitone` | iff | True | True | tactic_gap | `simp [Antitone, AntitoneOn]` | `simp [Antitone, AntitoneOn]` |
| `Set.ssubset_singleton_iff` | iff | True | True | search_depth_gap | `rw [ssubset_iff_subset_ne, subset_single` | `rw [ssubset_iff_subset_ne, sub` |
| `Set.ite_empty_right` | membership | True | True | tactic_gap | `simp [Set.ite]` | `simp [Set.ite]` |
| `Set.ite_inter` | membership | True | True | search_depth_gap | `rw [ite_inter_inter, ite_same]` | `rw [ite_inter_inter, ite_same]` |
| `Set.ite_inter_self` | membership | True | True | search_depth_gap | `rw [Set.ite, union_inter_distrib_right, ` | `rw [Set.ite, union_inter_distr` |
| `Set.ite_right` | membership | True | True | tactic_gap | `simp [Set.ite]` | `simp [Set.ite]` |

## `Set.diff_singleton_subset_iff`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u : Set α
x : α
s t : Set α
⊢ s \ {x} ⊆ t ↔ s ⊆ insert x t`
- classification: **search_depth_gap** | solved=True | win=`rw [← union_singleton, union_comm] <;> apply diff_subset_iff`
- minimality: [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'classical <;> aesop', 'solved': False}]

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all` | simp_all made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp [Set.subset_def]` |  |
| D_subset_diff_union | proof_failed | False | none | `simp_all [Set.subset_def]` |  |
| E_ite_bycases | proof_failed | False | none | `simp [Set.ite]` | simp made no progress |
| E_ite_bycases | proof_failed | False | none | `simp_all [Set.ite]` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp only [Set.ite]` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp [Set.ext_iff]` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all [Set.ext_iff]` | simp_all made no progress |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp_all` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp` | tactic 'apply' failed, failed to unify   ?a = ?b w |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp_a` | tactic 'apply' failed, failed to unify   ?a = ?b w |
| E_ite_bycases | parse_error | False | none | `classical <;> simp [Set.ite]` | <stdin>:1:10: expected '{' or tactic |
| A_baselines | proof_failed | False | none | `aesop` |  |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp` | applyExtTheorem only applies to equations, not   s |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp_all` | applyExtTheorem only applies to equations, not   s |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp [Set.ite]` | applyExtTheorem only applies to equations, not   s |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp_all [Set.ite]` | applyExtTheorem only applies to equations, not   s |
| E_ite_bycases | ext_not_applicable | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp_all [Set.` | applyExtTheorem only applies to equations, not   s |
| E_ite_bycases | ext_not_applicable | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp [Set.ite]` | applyExtTheorem only applies to equations, not   s |
| F_source_inspired | solved | True | high | `rw [← union_singleton, union_comm] <;> apply dif` |  |

## `Set.ite_eq_of_subset_left`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ : Set α
h : s₁ ⊆ s₂
⊢ t.ite s₁ s₂ = s₁ ∪ s₂ \ t`
- classification: **search_depth_gap** | solved=True | win=`ext x <;> by_cases hx : x ∈ t <;> simp [hx, Set.ite, or_iff_right_of_imp (@h x)]`
- minimality: [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'classical <;> aesop', 'solved': False}]

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all` | simp_all made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp [Set.subset_def]` | simp made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp_all [Set.subset_def]` |  |
| E_ite_bycases | proof_failed | False | none | `simp [Set.ite]` |  |
| E_ite_bycases | proof_failed | False | none | `simp_all [Set.ite]` |  |
| E_ite_bycases | proof_failed | False | none | `simp only [Set.ite]` |  |
| A_baselines | proof_failed | False | none | `simp [Set.ext_iff]` |  |
| A_baselines | proof_failed | False | none | `simp_all [Set.ext_iff]` |  |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp_all` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp` |  |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp_a` |  |
| E_ite_bycases | parse_error | False | none | `classical <;> simp [Set.ite]` | <stdin>:1:10: expected '{' or tactic |
| A_baselines | proof_failed | False | none | `aesop` | tactic 'aesop' failed, made no progress Initial go |
| B_ext_equality | proof_failed | False | none | `ext x <;> simp` |  |
| B_ext_equality | proof_failed | False | none | `ext x <;> simp_all` |  |
| B_ext_equality | proof_failed | False | none | `ext x <;> simp [Set.ite]` |  |
| B_ext_equality | proof_failed | False | none | `ext x <;> simp_all [Set.ite]` |  |
| E_ite_bycases | proof_failed | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp_all [Set.` |  |
| E_ite_bycases | proof_failed | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp [Set.ite]` |  |
| F_source_inspired | solved | True | high | `ext x <;> by_cases hx : x ∈ t <;> simp [hx, Set.` |  |

## `Set.pair_eq_pair_iff`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s s₁ s₂ t t₁ t₂ u : Set α
x y z w : α
⊢ {x, y} = {z, w} ↔ x = z ∧ y = w ∨ x = w ∧ y = z`
- classification: **tactic_gap** | solved=True | win=`simp [subset_antisymm_iff, insert_subset_iff] <;> aesop`
- minimality: [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'classical <;> aesop', 'solved': False}]

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all` | simp_all made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp [Set.subset_def]` | simp made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp_all [Set.subset_def]` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp [Set.ite]` | simp made no progress |
| E_ite_bycases | proof_failed | False | none | `simp_all [Set.ite]` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp only [Set.ite]` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp [Set.ext_iff]` |  |
| A_baselines | proof_failed | False | none | `simp_all [Set.ext_iff]` |  |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp_all` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp` | tactic 'apply' failed, failed to unify   ?a = ?b w |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp_a` | tactic 'apply' failed, failed to unify   ?a = ?b w |
| E_ite_bycases | parse_error | False | none | `classical <;> simp [Set.ite]` | <stdin>:1:10: expected '{' or tactic |
| A_baselines | proof_failed | False | none | `aesop` |  |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp` | applyExtTheorem only applies to equations, not   { |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp_all` | applyExtTheorem only applies to equations, not   { |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp [Set.ite]` | applyExtTheorem only applies to equations, not   { |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp_all [Set.ite]` | applyExtTheorem only applies to equations, not   { |
| E_ite_bycases | ext_not_applicable | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp_all [Set.` | applyExtTheorem only applies to equations, not   { |
| E_ite_bycases | ext_not_applicable | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp [Set.ite]` | applyExtTheorem only applies to equations, not   { |
| A_baselines | parse_error | False | none | `classical <;> aesop` | <stdin>:1:10: expected '{' or tactic |
| F_source_inspired | solved | True | medium | `simp [subset_antisymm_iff, insert_subset_iff] <;` |  |

## `Set.subset_insert_iff`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
x : α
⊢ s ⊆ insert x t ↔ s ⊆ t ∨ x ∈ s ∧ s \ {x} ⊆ t`
- classification: **needs_deeper_search** | solved=False | win=`None`

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` |  |
| A_baselines | proof_failed | False | none | `simp_all` |  |
| D_subset_diff_union | proof_failed | False | none | `simp [Set.subset_def]` |  |
| D_subset_diff_union | proof_failed | False | none | `simp_all [Set.subset_def]` |  |
| E_ite_bycases | unknown_ident | False | none | `simp [Set.ite]` | unknown constant 'Set.ite' |
| E_ite_bycases | unknown_ident | False | none | `simp_all [Set.ite]` | unknown constant 'Set.ite' |
| E_ite_bycases | proof_failed | False | none | `simp only [Set.ite]` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp [Set.ext_iff]` |  |
| A_baselines | proof_failed | False | none | `simp_all [Set.ext_iff]` |  |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp_all` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp` | tactic 'apply' failed, failed to unify   ?a = ?b w |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp_a` | tactic 'apply' failed, failed to unify   ?a = ?b w |
| E_ite_bycases | parse_error | False | none | `classical <;> simp [Set.ite]` | <stdin>:1:10: expected '{' or tactic |
| A_baselines | proof_failed | False | none | `aesop` |  |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp` | applyExtTheorem only applies to equations, not   s |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp_all` | applyExtTheorem only applies to equations, not   s |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp [Set.ite]` | applyExtTheorem only applies to equations, not   s |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp_all [Set.ite]` | applyExtTheorem only applies to equations, not   s |
| E_ite_bycases | ext_not_applicable | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp_all [Set.` | applyExtTheorem only applies to equations, not   s |
| E_ite_bycases | ext_not_applicable | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp [Set.ite]` | applyExtTheorem only applies to equations, not   s |
| A_baselines | parse_error | False | none | `classical <;> aesop` | <stdin>:1:10: expected '{' or tactic |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> aesop` | applyExtTheorem only applies to equations, not   s |
| F_source_inspired | proof_failed | False | high | `rw [← diff_singleton_subset_iff] <;> aesop` |  |

## `Set.subset_singleton_iff_eq`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s : Set α
x : α
⊢ s ⊆ {x} ↔ s = ∅ ∨ s = {x}`
- classification: **needs_deeper_search** | solved=False | win=`None`

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` |  |
| A_baselines | proof_failed | False | none | `simp_all` |  |
| D_subset_diff_union | proof_failed | False | none | `simp [Set.subset_def]` |  |
| D_subset_diff_union | proof_failed | False | none | `simp_all [Set.subset_def]` |  |
| E_ite_bycases | unknown_ident | False | none | `simp [Set.ite]` | unknown constant 'Set.ite' |
| E_ite_bycases | unknown_ident | False | none | `simp_all [Set.ite]` | unknown constant 'Set.ite' |
| E_ite_bycases | proof_failed | False | none | `simp only [Set.ite]` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp [Set.ext_iff]` |  |
| A_baselines | proof_failed | False | none | `simp_all [Set.ext_iff]` |  |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp_all` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp` | tactic 'apply' failed, failed to unify   ?a = ?b w |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp_a` | tactic 'apply' failed, failed to unify   ?a = ?b w |
| E_ite_bycases | parse_error | False | none | `classical <;> simp [Set.ite]` | <stdin>:1:10: expected '{' or tactic |
| F_source_inspired | proof_failed | False | medium | `constructor <;> intro h <;> simp_all` |  |
| A_baselines | proof_failed | False | none | `aesop` |  |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp` | applyExtTheorem only applies to equations, not   s |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp_all` | applyExtTheorem only applies to equations, not   s |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp [Set.ite]` | applyExtTheorem only applies to equations, not   s |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> simp_all [Set.ite]` | applyExtTheorem only applies to equations, not   s |
| E_ite_bycases | ext_not_applicable | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp_all [Set.` | applyExtTheorem only applies to equations, not   s |
| E_ite_bycases | ext_not_applicable | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp [Set.ite]` | applyExtTheorem only applies to equations, not   s |
| F_source_inspired | unknown_ident | False | high | `rcases s.eq_empty_or_nonempty with rfl | hs <;> ` | unknown identifier 'hs' |
| A_baselines | parse_error | False | none | `classical <;> aesop` | <stdin>:1:10: expected '{' or tactic |
| B_ext_equality | ext_not_applicable | False | none | `ext x <;> aesop` | applyExtTheorem only applies to equations, not   s |

## `Set.union_empty_iff`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
⊢ s ∪ t = ∅ ↔ s = ∅ ∧ t = ∅`
- classification: **tactic_gap** | solved=True | win=`simp only [← subset_empty_iff, union_subset_iff]`
- minimality: [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'classical <;> aesop', 'solved': False}]

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all` | simp_all made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp [Set.subset_def]` | simp made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp_all [Set.subset_def]` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp [Set.ite]` | simp made no progress |
| E_ite_bycases | proof_failed | False | none | `simp_all [Set.ite]` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp only [Set.ite]` | simp made no progress |
| F_source_inspired | solved | True | medium | `simp only [← subset_empty_iff, union_subset_iff]` |  |

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
- classification: **tactic_gap** | solved=True | win=`simp [Antitone, AntitoneOn]`
- minimality: [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'classical <;> aesop', 'solved': False}]

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp [Set.ite]` | simp made no progress |
| E_ite_bycases | proof_failed | False | none | `simp_all [Set.ite]` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp only [Set.ite]` | simp made no progress |
| F_source_inspired | solved | True | low | `simp [Antitone, AntitoneOn]` |  |

## `Set.ssubset_singleton_iff`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t t₁ t₂ u s : Set α
x : α
⊢ s ⊂ {x} ↔ s = ∅`
- classification: **search_depth_gap** | solved=True | win=`rw [ssubset_iff_subset_ne, subset_singleton_iff_eq, or_and_right, and_not_self_iff, or_false_iff, and_iff_left_iff_imp] <;> exact fun h => h ▸ (singleton_ne_empty _).symm`
- minimality: [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'classical <;> aesop', 'solved': False}]

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp [Set.ite]` | simp made no progress |
| E_ite_bycases | proof_failed | False | none | `simp_all [Set.ite]` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp only [Set.ite]` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp [Set.ext_iff]` |  |
| A_baselines | proof_failed | False | none | `simp_all [Set.ext_iff]` |  |
| C_iff_decomp | proof_failed | False | none | `constructor <;> intro h <;> simp_all` | simp_all made no progress |
| C_iff_decomp | proof_failed | False | none | `constructor <;> intro h <;> try simp_all` |  |
| C_iff_decomp | proof_failed | False | none | `refine ⟨?_, ?_⟩ <;> intro h <;> simp_all` | simp_all made no progress |
| E_ite_bycases | parse_error | False | none | `classical <;> simp [Set.ite]` | <stdin>:1:10: expected '{' or tactic |
| A_baselines | proof_failed | False | none | `aesop` |  |
| E_ite_bycases | ext_not_applicable | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp_all [Set.` | applyExtTheorem only applies to equations, not   s |
| E_ite_bycases | ext_not_applicable | False | none | `ext x <;> by_cases hx : x ∈ t <;> simp [Set.ite]` | applyExtTheorem only applies to equations, not   s |
| F_source_inspired | solved | True | high | `rw [ssubset_iff_subset_ne, subset_singleton_iff_` |  |

## `Set.ite_empty_right`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s : Set α
⊢ t.ite s ∅ = s ∩ t`
- classification: **tactic_gap** | solved=True | win=`simp [Set.ite]`
- minimality: [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'classical <;> aesop', 'solved': False}]

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all` | simp_all made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp [Set.subset_def]` | simp made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp_all [Set.subset_def]` | simp_all made no progress |
| E_ite_bycases | solved | True | none | `simp [Set.ite]` |  |

## `Set.ite_inter`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ s : Set α
⊢ t.ite (s₁ ∩ s) (s₂ ∩ s) = t.ite s₁ s₂ ∩ s`
- classification: **search_depth_gap** | solved=True | win=`rw [ite_inter_inter, ite_same]`
- minimality: [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'classical <;> aesop', 'solved': False}]

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all` | simp_all made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp [Set.subset_def]` | simp made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp_all [Set.subset_def]` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp [Set.ite]` |  |
| E_ite_bycases | proof_failed | False | none | `simp_all [Set.ite]` |  |
| E_ite_bycases | proof_failed | False | none | `simp only [Set.ite]` |  |
| A_baselines | proof_failed | False | none | `simp [Set.ext_iff]` |  |
| A_baselines | proof_failed | False | none | `simp_all [Set.ext_iff]` |  |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp_all` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp` |  |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp_a` |  |
| E_ite_bycases | parse_error | False | none | `classical <;> simp [Set.ite]` | <stdin>:1:10: expected '{' or tactic |
| F_source_inspired | solved | True | high | `rw [ite_inter_inter, ite_same]` |  |

## `Set.ite_inter_self`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s s' : Set α
⊢ t.ite s s' ∩ t = s ∩ t`
- classification: **search_depth_gap** | solved=True | win=`rw [Set.ite, union_inter_distrib_right, diff_inter_self, inter_assoc, inter_self, union_empty]`
- minimality: [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'classical <;> aesop', 'solved': False}]

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all` | simp_all made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp [Set.subset_def]` | simp made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp_all [Set.subset_def]` | simp_all made no progress |
| E_ite_bycases | proof_failed | False | none | `simp [Set.ite]` |  |
| E_ite_bycases | proof_failed | False | none | `simp_all [Set.ite]` |  |
| E_ite_bycases | proof_failed | False | none | `simp only [Set.ite]` |  |
| F_source_inspired | proof_failed | False | medium | `simp [Set.ite, inter_assoc]` |  |
| A_baselines | proof_failed | False | none | `simp [Set.ext_iff]` |  |
| A_baselines | proof_failed | False | none | `simp_all [Set.ext_iff]` |  |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `intro x <;> simp_all` | tactic 'introN' failed, insufficient number of bin |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp` |  |
| D_subset_diff_union | proof_failed | False | none | `apply Set.Subset.antisymm <;> intro x <;> simp_a` |  |
| E_ite_bycases | parse_error | False | none | `classical <;> simp [Set.ite]` | <stdin>:1:10: expected '{' or tactic |
| F_source_inspired | solved | True | high | `rw [Set.ite, union_inter_distrib_right, diff_int` |  |

## `Set.ite_right`
- initial goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u s t : Set α
⊢ s.ite t s = t ∩ s`
- classification: **tactic_gap** | solved=True | win=`simp [Set.ite]`
- minimality: [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'classical <;> aesop', 'solved': False}]

| family | outcome | solved | risk | probe | error |
|---|---|---|---|---|---|
| A_baselines | proof_failed | False | none | `simp` | simp made no progress |
| A_baselines | proof_failed | False | none | `simp_all` | simp_all made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp [Set.subset_def]` | simp made no progress |
| D_subset_diff_union | proof_failed | False | none | `simp_all [Set.subset_def]` | simp_all made no progress |
| E_ite_bycases | solved | True | none | `simp [Set.ite]` |  |

> Live LeanDojo probes. No solve is a confirmed win; NS23 minimal-sufficient relabel + deterministic reproduction required before any promotion. RC1/production configs untouched.