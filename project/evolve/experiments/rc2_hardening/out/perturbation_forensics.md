# RC2 Hardening — Perturbation-Win Forensics (+4)

- credit-status histogram: `{'sx3_sequence_candidate': 4}`
- SX3 sequence candidates: ['Set.ite_inter', 'Set.ite_inter_self', 'Set.ite_compl', 'Set.ite_inter_compl_self']
- excluded (search-order artifacts): []

| theorem | single-shot simp[Set.ite] | simp[Set.ite]<;>aesop | bare aesop | simp_all | role | credit |
|---|---|---|---|---|---|---|
| `Set.ite_inter` | proof_failed | solved | proof_failed | proof_failed | enabling_step | **sx3_sequence_candidate** |
| `Set.ite_inter_self` | proof_failed | solved | proof_failed | proof_failed | enabling_step | **sx3_sequence_candidate** |
| `Set.ite_compl` | proof_failed | solved | proof_failed | proof_failed | enabling_step | **sx3_sequence_candidate** |
| `Set.ite_inter_compl_self` | proof_failed | solved | proof_failed | proof_failed | enabling_step | **sx3_sequence_candidate** |

## `Set.ite_inter`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁✝ s₂✝ t✝ t₁ t₂ u t s₁ s₂ s : Set α
⊢ t.ite (s₁ ∩ s) (s₂ ∩ s) = t.ite s₁ s₂ ∩ s`
- candidate_role: **enabling_step** | credit: **sx3_sequence_candidate**
- reason: simp [Set.ite] then aesop/simp_all closes it (depth-2 sequence), but bare baselines and single-shot simp[Set.ite] do NOT -> SX3 sequence candidate, not an RC2 single-shot credited win
- RC2 finishing step: (2, 'aesop') | simp[Set.ite] advances in RC2 trace: True
- direct probes: {'simp [Set.ite]': 'proof_failed', 'simp [Set.ite] <;> aesop': 'solved', 'simp [Set.ite] <;> simp_all': 'proof_failed', 'simp [Set.ite] <;> try aesop': 'solved', 'aesop': 'proof_failed', 'simp_all': 'proof_failed', 'simp [Set.ext_iff]': 'proof_failed', 'simp [Set.ite, Set.ext_iff]': 'proof_failed', 'ext x <;> simp [Set.ite]': 'proof_failed'}

## `Set.ite_inter_self`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s s' : Set α
⊢ t.ite s s' ∩ t = s ∩ t`
- candidate_role: **enabling_step** | credit: **sx3_sequence_candidate**
- reason: simp [Set.ite] then aesop/simp_all closes it (depth-2 sequence), but bare baselines and single-shot simp[Set.ite] do NOT -> SX3 sequence candidate, not an RC2 single-shot credited win
- RC2 finishing step: (2, 'aesop') | simp[Set.ite] advances in RC2 trace: True
- direct probes: {'simp [Set.ite]': 'proof_failed', 'simp [Set.ite] <;> aesop': 'solved', 'simp [Set.ite] <;> simp_all': 'proof_failed', 'simp [Set.ite] <;> try aesop': 'solved', 'aesop': 'proof_failed', 'simp_all': 'proof_failed', 'simp [Set.ext_iff]': 'proof_failed', 'simp [Set.ite, Set.ext_iff]': 'proof_failed', 'ext x <;> simp [Set.ite]': 'proof_failed'}

## `Set.ite_compl`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s s' : Set α
⊢ tᶜ.ite s s' = t.ite s' s`
- candidate_role: **enabling_step** | credit: **sx3_sequence_candidate**
- reason: simp [Set.ite] then aesop/simp_all closes it (depth-2 sequence), but bare baselines and single-shot simp[Set.ite] do NOT -> SX3 sequence candidate, not an RC2 single-shot credited win
- RC2 finishing step: None | simp[Set.ite] advances in RC2 trace: True
- direct probes: {'simp [Set.ite]': 'proof_failed', 'simp [Set.ite] <;> aesop': 'solved', 'simp [Set.ite] <;> simp_all': 'proof_failed', 'simp [Set.ite] <;> try aesop': 'solved', 'aesop': 'proof_failed', 'simp_all': 'proof_failed', 'simp [Set.ext_iff]': 'proof_failed', 'simp [Set.ite, Set.ext_iff]': 'proof_failed', 'ext x <;> simp [Set.ite]': 'proof_failed'}

## `Set.ite_inter_compl_self`
- goal: `α : Type u
β : Type v
γ : Type w
ι : Sort x
a b : α
s✝ s₁ s₂ t✝ t₁ t₂ u t s s' : Set α
⊢ t.ite s s' ∩ tᶜ = s' ∩ tᶜ`
- candidate_role: **enabling_step** | credit: **sx3_sequence_candidate**
- reason: simp [Set.ite] then aesop/simp_all closes it (depth-2 sequence), but bare baselines and single-shot simp[Set.ite] do NOT -> SX3 sequence candidate, not an RC2 single-shot credited win
- RC2 finishing step: (2, 'aesop') | simp[Set.ite] advances in RC2 trace: True
- direct probes: {'simp [Set.ite]': 'proof_failed', 'simp [Set.ite] <;> aesop': 'solved', 'simp [Set.ite] <;> simp_all': 'proof_failed', 'simp [Set.ite] <;> try aesop': 'solved', 'aesop': 'proof_failed', 'simp_all': 'proof_failed', 'simp [Set.ext_iff]': 'proof_failed', 'simp [Set.ite, Set.ext_iff]': 'proof_failed', 'ext x <;> simp [Set.ite]': 'proof_failed'}
