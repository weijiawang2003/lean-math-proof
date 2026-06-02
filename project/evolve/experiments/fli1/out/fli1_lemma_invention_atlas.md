# FLI1 Lemma-Invention Atlas

_From captured residual goals to candidate intermediate lemmas, proofs, and downstream rescue. Hedged language: candidates *suggest* missing lemmas; rescues are the real test._

## 1. Overview
FLI1 re-ran the 40 FLI0 seed failures live, captured residual goals, synthesized candidate intermediate lemmas, checked existence, typechecked, proved, and tested downstream rescue at each theorem's file position.

## 2. Residual goal capture
- captured 40/40 (high-quality 19); 0 solved directly.

## 3. Goal clusters
- 21 clusters by (namespace, pattern, relation, container-op).

## 4. Candidate lemma families
- 40 candidates synthesized (one per captured seed).

## 5. Existing lemma / retrieval gaps
- existing-check: {'PROBABLY_NEW': 21, 'EXISTS_CLOSE': 15, 'TOO_VAGUE_TO_CHECK': 4}
- **retrieval gaps (close lemma was retrieved but search didn't use it): 15**

## 6. Typechecking results
- {'TYPECHECKS': 22, 'TYPE_ERROR': 8, 'UNIVERSE_OR_TYPECLASS_ERROR': 10}

## 7. Candidate lemma proof results
- PROVED: 1

## 8. Downstream rescue attempts
- {'NO_RESCUE': 38, 'DOWNSTREAM_RESCUE': 1, 'DIRECT_SOLVE_DUPLICATE': 1}
- **DOWNSTREAM_RESCUE: 1**

## 9. Best examples

### FLI1-L04 — `Finset.card_le_one_iff` (DOWNSTREAM_RESCUE)
- residual goal: `case mp / α : Type u_1 / β : Type u_2 / R : Type u_3 / s t u : Finset α / f : α → β / n : ℕ / h : s.card ≤ 1 / ⊢ ∀ {a b : α}, a ∈ s → b ∈ s → a = b /  / case mpr / α : Type u_1 / β : Type u_2 / R : Type u_3 / s t u : Finset α / f `
- existing check: EXISTS_CLOSE (closest `Finset.card_le_one`)
- typecheck: TYPECHECKS | proved: FAILED via `None`
- rescue: **DOWNSTREAM_RESCUE** via `simp [Finset.card_le_one] <;> aesop` (robust=True)
- At `Finset.card_le_one_iff`'s position the restricted search failed; existing lemma `Finset.card_le_one` closes the goal via `simp [Finset.card_le_one] <;> aesop`.

### FLI1-L02 — `Finset.card_le_card` (NO_RESCUE)
- residual goal: `α : Type u_1 / β : Type u_2 / R : Type u_3 / s t : Finset α / a b : α / h : s ⊆ t / ⊢ s.card ≤ t.card`
- existing check: PROBABLY_NEW (closest `Finset.eq_of_subset_of_card_le`)
- typecheck: TYPECHECKS | proved: PROVED via `gcongr`
- rescue: **NO_RESCUE** via `None` (robust=None)
- At `Finset.card_le_card`'s position the restricted search failed; the synthesized intermediate lemma simplifies the goal via `None`.

## 10. Failure modes

- 18/40 candidate statements fail to typecheck (universe/typeclass/type errors from reconstructing binders out of pretty-printed goal context).
- Many residual goals fold the hypotheses into binders, so the standalone goal can look trivial; the content lives in the binders.
- Rescue at-position is the honest bar: a candidate that proves under full `import Module` may still not rescue at the theorem's position.

## 11. Recommendations for FLI2

1. Fix candidate-statement synthesis (carry exact binder types from the goal's local context rather than pp reconstruction; preserve distinct universes/instances).
2. Prioritize the retrieval-gap cluster — these bridges *already exist*; the win is a routing/deployment fix (gated `simp [L]`), exactly the RC4B/RC4C pattern.
3. For PARTIAL_PROGRESS cases, capture the new residual after the candidate and iterate (multi-step invention).
4. Promote nothing into production; keep FLI as an off-line discovery track.
