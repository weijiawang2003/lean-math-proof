# FLI2 Retrieval-Gap Deployment Atlas

_Can failure analysis turn retrieved-but-undeployed lemmas into reusable deployment rules that rescue failed theorems? Hedged language; at-position tests only._

## 1. Overview
Pool 217 retrieval-gap/high-signal failures → 1472 gated deployment actions → 1059 run live over 161 theorems.

## 2. Why FLI2 follows FLI1
FLI1 found 1 robust rescue and 15 retrieval gaps — the relevant lemma often already exists but is not deployed. FLI2 scales the test: do retrieval gaps form reusable deployment families?

## 3. Retrieval-gap pool
217 cases (FLI1 retrieval-gap + FLI0 high-signal), namespaces {'Finset': 66, 'Set': 44, 'List': 41, 'Multiset': 35, 'Nat': 31}.

## 4. Deployment action families
{'SIMPLE_SIMP': 617, 'SIMP_AESOP': 502, 'EXACT_LEMMA': 188, 'CONSTRUCTOR_SIMP': 76, 'OMEGA_CLOSER': 46, 'GCONGR_CLOSER': 38, 'EXT_SIMP': 3, 'INTRO_SIMP_AESOP': 2}

## 5. Live evaluation results
action classifications: {'NO_RESCUE': 880, 'UNKNOWN_NAME_OR_IMPORT_GAP': 101, 'PARTIAL_PROGRESS': 64, 'TRUE_RETRIEVAL_GAP_RESCUE': 11, 'CONTROL_DUPLICATE': 2, 'NEEDS_REVIEW': 1}

## 6. True retrieval-gap rescues

| theorem | lemma | tactic |
|---|---|---|
| `Finset.card_le_one_iff` | `Finset.card_le_one` | `simp [Finset.card_le_one] <;> aesop` |
| `Finset.mem_filterMap` | `Finset.filterMap` | `simp [Finset.filterMap]` |
| `Finset.mem_filterMap` | `Finset.filterMap` | `simp [Finset.filterMap] <;> aesop` |
| `Finset.card_subtype` | `Finset.subtype` | `simp [Finset.subtype]` |
| `Finset.card_subtype` | `Finset.subtype` | `simp [Finset.subtype] <;> aesop` |
| `Finset.mem_map` | `Finset.map` | `simp [Finset.map]` |
| `Finset.mem_map` | `Finset.map` | `simp [Finset.map] <;> aesop` |
| `Finset.mem_preimage` | `Finset.preimage` | `simp [Finset.preimage]` |
| `Finset.mem_preimage` | `Finset.preimage` | `simp [Finset.preimage] <;> aesop` |
| `List.bidirectionalRec_singleton` | `List.bidirectionalRec` | `simp [List.bidirectionalRec]` |
| `List.bidirectionalRec_singleton` | `List.bidirectionalRec` | `simp [List.bidirectionalRec] <;> aesop` |

## 7. Partial progress cases

### `Finset.card_le_one_iff` via `simp [Finset.card_le_one]`
- before: `α : Type u_1
β : Type u_2
R : Type u_3
s t u : Finset α
f : α → β
n : ℕ
⊢ s.card ≤ 1 ↔ ∀ {a b : α}, a ∈ s → b ∈ s → a = `
- after:  `α : Type u_1
β : Type u_2
R : Type u_3
s t u : Finset α
f : α → β
n : ℕ
⊢ (∀ a ∈ s, ∀ b ∈ s, a = b) ↔ ∀ {a b : α}, a ∈ s`

### `Finset.card_le_one_iff` via `simp [Finset.card_le_one]`
- before: `α : Type u_1
β : Type u_2
R : Type u_3
s t u : Finset α
f : α → β
n : ℕ
⊢ s.card ≤ 1 ↔ ∀ {a b : α}, a ∈ s → b ∈ s → a = `
- after:  `α : Type u_1
β : Type u_2
R : Type u_3
s t u : Finset α
f : α → β
n : ℕ
⊢ (∀ a ∈ s, ∀ b ∈ s, a = b) ↔ ∀ {a b : α}, a ∈ s`

### `Finset.image_val_of_injOn` via `simp [Finset.image_val]`
- before: `α : Type u_1
β : Type u_2
γ : Type u_3
inst✝ : DecidableEq β
f g : α → β
s : Finset α
t : Finset β
a : α
b c : β
H : Set`
- after:  `α : Type u_1
β : Type u_2
γ : Type u_3
inst✝ : DecidableEq β
f g : α → β
s : Finset α
t : Finset β
a : α
b c : β
H : Set`

### `Finset.card_filter_le` via `gcongr`
- before: `α : Type u_1
β : Type u_2
R : Type u_3
s✝ t u : Finset α
f : α → β
n : ℕ
s : Finset α
p : α → Prop
inst✝ : DecidablePred`
- after:  `case a
α : Type u_1
β : Type u_2
R : Type u_3
s✝ t u : Finset α
f : α → β
n : ℕ
s : Finset α
p : α → Prop
inst✝ : Decida`

### `Finset.card_le_one_iff_subsingleton_coe` via `simp [Finset.card_le_one]`
- before: `α : Type u_1
β : Type u_2
R : Type u_3
s t u : Finset α
f : α → β
n : ℕ
⊢ s.card ≤ 1 ↔ Subsingleton { x // x ∈ s }`
- after:  `α : Type u_1
β : Type u_2
R : Type u_3
s t u : Finset α
f : α → β
n : ℕ
⊢ (∀ a ∈ s, ∀ b ∈ s, a = b) ↔ Subsingleton { x /`

### `Finset.card_le_one_of_subsingleton` via `simp [Finset.card_le_one_iff_subsingleton_coe]`
- before: `α : Type u_1
β : Type u_2
R : Type u_3
s✝ t u : Finset α
f : α → β
n : ℕ
inst✝ : Subsingleton α
s : Finset α
⊢ s.card ≤ `
- after:  `α : Type u_1
β : Type u_2
R : Type u_3
s✝ t u : Finset α
f : α → β
n : ℕ
inst✝ : Subsingleton α
s : Finset α
⊢ Subsingle`

### `Finset.card_le_one_of_subsingleton` via `simp [Finset.card_le_one]`
- before: `α : Type u_1
β : Type u_2
R : Type u_3
s✝ t u : Finset α
f : α → β
n : ℕ
inst✝ : Subsingleton α
s : Finset α
⊢ s.card ≤ `
- after:  `α : Type u_1
β : Type u_2
R : Type u_3
s✝ t u : Finset α
f : α → β
n : ℕ
inst✝ : Subsingleton α
s : Finset α
⊢ ∀ a ∈ s, `

### `Finset.map_val_val_powersetCard` via `simp [Finset.powersetCard]`
- before: `α : Type u_1
s✝¹ t✝ : Finset α
n : ?m.17539
s✝ t s : Finset α
i : ℕ
⊢ Multiset.map val (powersetCard i s).val = Multiset`
- after:  `α : Type u_1
s✝¹ t✝ : Finset α
n : ?m.17539
s✝ t s : Finset α
i : ℕ
⊢ Multiset.map val (pmap mk (Multiset.powersetCard i`

## 8. Deployment rules mined

| rule | family | actions | rescues | partials | FP | status |
|---|---|---|---|---|---|---|
| FINSET_MAP_BRIDGE | Finset.map_* | SIMPLE_SIMP,SIMP_AESOP | 4 | 0 | 45 | needs_more_data |
| FINSET_IMAGE_BRIDGE | Finset.image_* | SIMPLE_SIMP,SIMP_AESOP | 2 | 1 | 6 | needs_more_data |
| FINSET_SUBTYPE_BRIDGE | Finset.subtype_* | SIMPLE_SIMP,SIMP_AESOP | 2 | 0 | 0 | candidate |
| LIST_BIDIRECTIONALREC_BRIDGE | List.bidirectionalrec_* | SIMPLE_SIMP,SIMP_AESOP | 2 | 0 | 6 | needs_more_data |
| FINSET_CARD_BRIDGE | Finset.card_* | SIMP_AESOP | 1 | 6 | 145 | needs_more_data |
| FINSET_MEM_BRIDGE | Finset.mem_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 15 | 21 | needs_more_data |
| LIST_SINGLETON_BRIDGE | List.singleton_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 6 | 6 | needs_more_data |
| MULTISET_MAP_BRIDGE | Multiset.map_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 5 | 85 | needs_more_data |
| SET_IMAGE_BRIDGE | Set.image_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 5 | 43 | needs_more_data |
| SET_SUBSET_BRIDGE | Set.subset_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 5 | 52 | needs_more_data |
| MULTISET_MEM_BRIDGE | Multiset.mem_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 4 | 13 | needs_more_data |
| FINSET_INSERT_BRIDGE | Finset.insert_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 2 | 4 | needs_more_data |
| MULTISET_BIND_BRIDGE | Multiset.bind_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 2 | 8 | needs_more_data |
| MULTISET_FILTER_BRIDGE | Multiset.filter_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 2 | 49 | needs_more_data |
| FINSET_BIUNION_BRIDGE | Finset.biunion_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 15 | needs_more_data |
| FINSET_CLOSER_BRIDGE | closer | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 32 | needs_more_data |
| FINSET_COE_BRIDGE | Finset.coe_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 1 | needs_more_data |
| FINSET_SUBSET_BRIDGE | Finset.subset_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 41 | needs_more_data |
| LIST_SUBSET_BRIDGE | List.subset_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 3 | needs_more_data |
| MULTISET_ATTACH_BRIDGE | Multiset.attach_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 1 | needs_more_data |
| MULTISET_CARD_BRIDGE | Multiset.card_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 5 | needs_more_data |
| MULTISET_COUNT_BRIDGE | Multiset.count_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 3 | needs_more_data |
| SET_IINTER_BRIDGE | Set.iinter_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 8 | needs_more_data |
| SET_IUNION_BRIDGE | Set.iunion_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 28 | needs_more_data |
| SET_LEFTINVON_BRIDGE | Set.leftinvon_* | SIMPLE_SIMP,SIMP_AESOP | 0 | 1 | 1 | needs_more_data |

## 9. Comparison to RC4B/RC4C

FLI2 discovers the same KIND of object RC4B/RC4C were hand-built for — a small gated `simp [L]`/closer action that deploys an existing lemma — but sourced automatically from failure analysis rather than manual curation. 11 at-position rescue(s) found; 0 overlap RC4-style families, 11 are new. Whether it becomes an RC-candidate generator depends on each rescue passing the full literal-RC2 additive validation (off-gate/floors/determinism) used for RC4A–RC4D; FLI2 only produces candidates, it does not validate or promote them.

## 10. Failure modes

- Many candidate solves are CONTROL_DUPLICATE: a bare control already closes the goal at position, i.e. RC5's failure does not reproduce in plain LeanDojo (its grammar was stricter / context differed).
- UNKNOWN_NAME: a retrieved lemma is not in scope at the theorem's file position (defined later) — an honest availability gap, not a deployable bridge.
- The deployable signal concentrates where a control genuinely fails but a specific retrieved lemma closes it.

## 11. Recommended next step (FLI3)

1. Promote confirmed deployment-rule candidates into the RC4-style literal-RC2 additive validation harness (off-gate/floors/determinism) — turn discovery into validated candidates.
2. For CONTROL_DUPLICATE cases, re-mine the FLI0 corpus distinguishing 'RC5-grammar gap' (control works at position) from 'genuine bridge needed'.
3. For PARTIAL_PROGRESS, feed the new residual to FLI1-style lemma invention (multi-step).
