# Source-Proof Analysis — `Multiset.toFinset_eq_singleton_iff` (corrected, live-verified)

- resolves for Dojo under **`Mathlib/Data/Finset/Basic.lean`**; proof text in the cache
  shard `Finset/Basic1th4x_l3.lean:2977` (this cache uses a sharded mathlib layout).
- statement (live-confirmed goal): `s.toFinset = {a} ↔ card s ≠ 0 ∧ s = card s • {a}`.

## Official proof (verbatim from cache)
```lean
theorem toFinset_eq_singleton_iff (s : Multiset α) (a : α) :
    s.toFinset = {a} ↔ card s ≠ 0 ∧ s = card s • {a} := by
  refine ⟨fun H ↦ ⟨fun h ↦ ?_, ext' fun x ↦ ?_⟩, fun H ↦ ?_⟩
  · rw [card_eq_zero.1 h, toFinset_zero] at H
    exact Finset.singleton_ne_empty _ H.symm
  · rw [count_nsmul, count_singleton]
    by_cases hx : x = a
    · simp_rw [hx, ite_true, mul_one, count_eq_card]
      intro y hy
      rw [← mem_toFinset, H, Finset.mem_singleton] at hy
      exact hy.symm
    have hx' : x ∉ s := fun h' ↦ hx <| by rwa [← mem_toFinset, H, Finset.mem_singleton] at h'
    simp_rw [count_eq_zero_of_not_mem hx', hx, ite_false, Nat.mul_zero]
  simpa only [toFinset_nsmul _ _ H.1, toFinset_singleton] using congr($(H.2).toFinset)
```
Tokens: `refine`×1, `ext'`×1, `by_cases`×1, `simp_rw`×2, `count_*` rewrites — and
**`induction`×0, `aesop`×0, `simp_all`×0**. It is a count-level extensionality proof.

## Dependency map — all dependencies EXIST
`card_eq_zero`, `toFinset_zero`, `Finset.singleton_ne_empty`, `count_nsmul`,
`count_singleton`, `count_eq_card`, `count_eq_zero_of_not_mem`, `mem_toFinset`,
`Finset.mem_singleton`, **`toFinset_nsmul`** (= holdout thm #1), `toFinset_singleton`.
⇒ **no missing lemma**; the gap is orchestration (split-iff → count-ext + by_cases).

## Why WX3 induction fails
WX3 `induction s using Multiset.induction_on <;> simp_all` is mis-applied: the RHS has `card s • {a}`, so cons-induction yields a more tangled nsmul/insert iff (confirmed in the RC1 trace residual) that simp_all cannot close. The official proof never inducts; it splits the iff and does count-extensionality.

## Live probe outcome (Part 3)
Part-3 ladder (13 probes, single live Dojo) closed it 0/13: 6 proof_failed, 6 max_recursion (every constructor<;>intro<;>simp_all[...] variant, INCLUDING the source-inspired simp set with toFinset_nsmul/toFinset_singleton/mem_toFinset/mem_singleton), 1 parse_error (run_transition rejects multi-line ·-bullet blocks, so the verbatim official proof can't be sent as one tactic). Confirms: this failure is NOT reachable by a one-shot simp/aesop-class probe; it needs the structured count-extensionality proof or a multi-step search.

## Classification
- **verdict:** reusable_probe (iff-split opener) + too_specialized (count-extensionality closure); NOT a missing lemma
- copying official proof would solve it: **True** (but it is depth ~4 with a case split)
- reusable probe: YES (opener only): `constructor`/`refine ⟨…⟩` to split the iff before any ext/membership step. Reusable across toFinset/Finset membership-iff goals.
- missing lemma: NO. Every dependency exists in Mathlib (card_eq_zero, count_*, mem_toFinset, toFinset_nsmul, toFinset_singleton, singleton_ne_empty).

## Implication for SF3
Do NOT invent a lemma. The honest SF3 levers are: (a) a SELECTIVITY/routing fix so the WX3 induction oracle does NOT fire on toFinset membership-iff goals (it strictly hurts here); (b) a MULTI-STEP search capability (the official proof is depth ~4 with a case split) rather than a single battery tactic — a one-shot probe cannot close it. No promotion: this remains an open genuine failure.