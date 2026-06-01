# SF2 Micro Failure Pattern Miner — Multiset holdout seed

- branch: `rc1-production-stack` | scope: analysis + live probes; no production promotion; no commit.

## 1. Executive summary

- **Ground-truth correction:** the run's `metrics.json` shows RC1 actually proved **2/3** of the Multiset holdout, with **1 genuine failure(s)** — NOT 0/3. The reported "0/3" was an **SF1 `eval_matrix` metrics-parser bug** (`parse_metrics` keyed on `proof_finished`/`solved`; the real metrics key is `finished`).
- Genuine RC1 failure(s): `Multiset.toFinset_eq_singleton_iff`.
- The other two are real RC1 **wins**: `Multiset.toFinset_nsmul` (`aesop`), `Multiset.disjoint_toFinset` (**WX3 Multiset induction oracle** — it fired *and closed*).
- Live LeanDojo probe runner available: `True`; any probe solved a goal: `True`.
- **No production promotion is made**; RC1 wrapper / NS24 router untouched.
- SF1↔metrics discrepancies (to fix in `scripts/sf1_eval_matrix.py`): [{'full_name': 'Multiset.toFinset_nsmul', 'sf1_eval_matrix_solved': False, 'metrics_finished': True}, {'full_name': 'Multiset.disjoint_toFinset', 'sf1_eval_matrix_solved': False, 'metrics_finished': True}]

## 2. Failure case table

| theorem | file_path | RC1 | winning tactic | WX3 fired | WX3 closed | ctx |
|---|---|---|---|---|---|---|
| `Multiset.toFinset_nsmul` | `Mathlib/Data/Finset/Basic.lean` | PROVED | `aesop` | False | False | full_trace |
| `Multiset.toFinset_eq_singleton_iff` | `Mathlib/Data/Finset/Basic.lean` | FAILED | `None` | True | False | full_trace |
| `Multiset.disjoint_toFinset` | `Mathlib/Data/Finset/Basic.lean` | PROVED | `induction m1 using Multiset.induction_on <;> simp_all` | True | True | full_trace |

### Genuine failure `Multiset.toFinset_eq_singleton_iff` — final residual goal
```lean
case h
α : Type u_1
β : Type u_2
γ : Type u_3
inst : DecidableEq α
s t : Multiset α
a : α
a_1 : ∅ = {a}
⊢ False

case cons
α : Type u_1
β : Type u_2
γ : Type u_3
inst✝ : DecidableEq α
s t : Multiset α
a a✝¹ : α
s✝ : Multiset α
a✝ : s✝.toFinset = {a} ↔ ¬s✝ = 0 ∧ s✝ = card s✝ • {a}
⊢ insert a✝¹ s✝.toFinset = {a} ↔ a✝¹ ::ₘ s✝ = (card s✝ + 1) • {a}
```
- final error: `All top-13 tactics errored at step 4`
- tactics RC1 tried: ['simp [Set.univ_union]', 'induction s using Multiset.induction_on <;> simp_all', 'aesop']

## 3. Source context summary

### `Multiset.toFinset_nsmul`
- source_found: `True` | path: `/Users/weijiawang/.cache/lean_dojo/leanprover-community-mathlib4-29dcec074de168ac2bf835a77ef68bbe069194c5/mathlib4/Mathlib/Data/Finset/Basic.lean` | decl line: 3125
```lean
theorem toFinset_nsmul (s : Multiset α) : ∀ n ≠ 0, (n • s).toFinset = s.toFinset
  | 0, h => by contradiction
  | n + 1, _ => by
```
- Mathlib proof pattern: `by_cases h : n = 0
    · rw [h, zero_add, one_nsmul]
    · rw [add_nsmul, toFinset_add, one_nsmul, toFinset_nsmul s n h, Finset.union_idempotent]
#align multiset.to_finset_nsmul Multiset.toFinset_nsmul

theorem toFinset_eq_singleton_iff (s `
- nearby toFinset/disjoint/nsmul/singleton lemmas: ['Finset.union_symm_inl', 'Finset.union_symm_inr', 'Nodup.toFinset_inj', 'Nonempty.exists_eq_singleton_or_nontrivial', 'Nonempty.subset_singleton_iff', 'Nontrivial.ne_singleton', 'Nontrivial.sdiff_singleton_nonempty', '_root_.Disjoint.forall_ne_finset', '_root_.Set.pairwiseDisjoint_filter', 'and', 'coe_disjUnion', 'coe_eq_singleton']

### `Multiset.toFinset_eq_singleton_iff`
- source_found: `True` | path: `/Users/weijiawang/.cache/lean_dojo/leanprover-community-mathlib4-29dcec074de168ac2bf835a77ef68bbe069194c5/mathlib4/Mathlib/Data/Finset/Basic.lean` | decl line: 3133
```lean
theorem toFinset_eq_singleton_iff (s : Multiset α) (a : α) :
    s.toFinset = {a} ↔ card s ≠ 0 ∧ s = card s • {a} := by
```
- Mathlib proof pattern: `refine ⟨fun H ↦ ⟨fun h ↦ ?_, ext' fun x ↦ ?_⟩, fun H ↦ ?_⟩
  · rw [card_eq_zero.1 h, toFinset_zero] at H
    exact Finset.singleton_ne_empty _ H.symm
  · rw [count_nsmul, count_singleton]
    by_cases hx : x = a
    · simp_rw [hx, ite_true,`
- nearby toFinset/disjoint/nsmul/singleton lemmas: ['Finset.union_symm_inl', 'Finset.union_symm_inr', 'Nodup.toFinset_inj', 'Nonempty.exists_eq_singleton_or_nontrivial', 'Nonempty.subset_singleton_iff', 'Nontrivial.ne_singleton', 'Nontrivial.sdiff_singleton_nonempty', '_root_.Disjoint.forall_ne_finset', '_root_.Set.pairwiseDisjoint_filter', 'and', 'coe_disjUnion', 'coe_eq_singleton']

### `Multiset.disjoint_toFinset`
- source_found: `True` | path: `/Users/weijiawang/.cache/lean_dojo/leanprover-community-mathlib4-29dcec074de168ac2bf835a77ef68bbe069194c5/mathlib4/Mathlib/Data/Finset/Basic.lean` | decl line: 3523
```lean
theorem disjoint_toFinset {m1 m2 : Multiset α} :
    _root_.Disjoint m1.toFinset m2.toFinset ↔ m1.Disjoint m2 := by
```
- Mathlib proof pattern: `rw [Finset.disjoint_iff_ne]
  refine ⟨fun h a ha1 ha2 => ?_, ?_⟩
  · rw [← Multiset.mem_toFinset] at ha1 ha2
    exact h _ ha1 _ ha2 rfl
  · rintro h a ha b hb rfl
    rw [Multiset.mem_toFinset] at ha hb
    exact h ha hb
#align multiset.di`
- nearby toFinset/disjoint/nsmul/singleton lemmas: ['Finset.union_symm_inl', 'Finset.union_symm_inr', 'Nodup.toFinset_inj', 'Nonempty.exists_eq_singleton_or_nontrivial', 'Nonempty.subset_singleton_iff', 'Nontrivial.ne_singleton', 'Nontrivial.sdiff_singleton_nonempty', '_root_.Disjoint.forall_ne_finset', '_root_.Set.pairwiseDisjoint_filter', 'and', 'coe_disjUnion', 'coe_eq_singleton']

## 4. Pattern analysis

- **toFinset membership bridge** (`Multiset.mem_toFinset`): the universal simp bridge for all three.
- **singleton bridge** (`Finset.eq_singleton_iff_unique_mem` ∘ membership): the one genuine failure `Multiset.toFinset_eq_singleton_iff`.
- **disjoint bridge** (`Finset.disjoint_left` ∘ membership): `Multiset.disjoint_toFinset` (RC1 win).
- **nsmul/toFinset bridge** (`Multiset.mem_nsmul`): `Multiset.toFinset_nsmul` (RC1 win).

**Key SF2 insight (sharpened by the real goal):** the one genuine failure is a membership/extensionality + singleton goal. RC1's WX3 `Multiset.induction_on` oracle FIRED on it and made it *worse* — the captured residual goal is `insert a✝¹ s✝.toFinset = {a} ↔ a✝¹ ::ₘ s✝ = (card s✝ + 1) • {a}`, i.e. induction+simp_all turned a clean singleton-iff into an `nsmul`-of-singleton tangle. The right family is `rw [Finset.eq_singleton_iff_unique_mem]; simp [Multiset.mem_toFinset]` (or `ext`+membership), NOT structural induction. This is the first concrete evidence that the WX3 induction oracle is *mis-applied* on toFinset-membership goals — exactly the SF2/SF3 lever.

## 5. Probe ladder

- generic base probes: ['simp', 'simp_all', 'aesop', 'classical; aesop', 'ext x <;> simp', 'ext x <;> simp_all', 'induction {var} using Multiset.induction_on <;> simp_all', 'induction {var} using Multiset.induction_on <;> simp [Multiset.toFinset_cons]', 'induction {var} using Multiset.induction_on <;> aesop']
- `Multiset.disjoint_toFinset` (motifs ['toFinset_membership_bridge', 'disjoint_bridge', 'multiset_finset_ext']): ['simp [Multiset.disjoint_toFinset]', 'simp [Disjoint, Multiset.disjoint_left, Multiset.mem_toFinset]', 'simp only [Finset.disjoint_left, Multiset.mem_toFinset]', 'rw [Finset.disjoint_left]; simp [Multiset.mem_toFinset]', 'constructor <;> intro h <;> simp_all [Finset.disjoint_left, Multiset.mem_toFinset, Multiset.disjoint_left]', 'aesop (add simp [Multiset.mem_toFinset, Finset.disjoint_left, Multiset.disjoint_left])']
- `Multiset.toFinset_eq_singleton_iff` (motifs ['singleton_bridge', 'multiset_finset_ext', 'toFinset_membership_bridge']): ['simp [Finset.eq_singleton_iff_unique_mem, Multiset.mem_toFinset]', 'rw [Finset.eq_singleton_iff_unique_mem]; simp [Multiset.mem_toFinset]', 'constructor <;> intro h <;> simp_all [Finset.ext_iff, Multiset.mem_toFinset]', 'ext x <;> simp [Multiset.mem_toFinset]', 'aesop (add simp [Multiset.mem_toFinset, Finset.eq_singleton_iff_unique_mem])']
- `Multiset.toFinset_nsmul` (motifs ['nsmul_toFinset_bridge', 'toFinset_membership_bridge', 'nat_induction']): ['ext x <;> simp [Multiset.mem_toFinset, Multiset.mem_nsmul]', 'ext x <;> simp [Multiset.mem_nsmul]', 'induction n <;> simp_all [Multiset.toFinset_cons, Multiset.succ_nsmul]', 'induction s using Multiset.induction_on <;> simp_all [Multiset.mem_nsmul]', 'simp [Finset.ext_iff, Multiset.mem_toFinset, Multiset.mem_nsmul]', 'aesop (add simp [Multiset.mem_toFinset, Multiset.mem_nsmul])']

## 6. Probe results

- `Multiset.toFinset_nsmul` (rc1_solved=True): solved_probe=aesop (add simp [Multiset.mem_toFinset, Multiset.mem_nsmul]); #solving_outcomes=1; run_error=None
  - minimality (unconfirmed, requires NS23 relabel): [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': True}, {'probe': 'ext x <;> simp', 'solved': False}]
  - solved=False `ext x <;> simp [Multiset.mem_toFinset, Multiset.mem_nsmul]`  err=`applyExtTheorem only applies to equations, not
  ∀ (n : ℕ), n ≠ 0 → (n • s).toFinset = s.toFinset`
  - solved=False `ext x <;> simp [Multiset.mem_nsmul]`  err=`applyExtTheorem only applies to equations, not
  ∀ (n : ℕ), n ≠ 0 → (n • s).toFinset = s.toFinset`
  - solved=False `induction n <;> simp_all [Multiset.toFinset_cons, Multiset.succ_nsmul]`  err=`tactic 'induction' failed, major premise type is not an inductive type 
  ?m.253804
α : Type u_1
β : Type u_2
γ : Type u_3
inst✝ : DecidableEq α
s✝ t s : Multiset α
x✝ : ?m.253804
⊢ ∀ (n : ℕ), n ≠ 0 →`
  - solved=False `induction s using Multiset.induction_on <;> simp_all [Multiset.mem_nsmul]`
  - solved=False `simp [Finset.ext_iff, Multiset.mem_toFinset, Multiset.mem_nsmul]`
  - solved=True `aesop (add simp [Multiset.mem_toFinset, Multiset.mem_nsmul])`
- `Multiset.toFinset_eq_singleton_iff` (rc1_solved=False): solved_probe=None; #solving_outcomes=0; run_error=None
  - solved=False `simp [Finset.eq_singleton_iff_unique_mem, Multiset.mem_toFinset]`
  - solved=False `rw [Finset.eq_singleton_iff_unique_mem]; simp [Multiset.mem_toFinset]`  err=`<stdin>:1:39: expected end of input`
  - solved=False `constructor <;> intro h <;> simp_all [Finset.ext_iff, Multiset.mem_toFinset]`  err=`tactic 'simp' failed, nested error:
maximum recursion depth has been reached
use `set_option maxRecDepth <num>` to increase limit
use `set_option diagnostics true` to get diagnostic information`
  - solved=False `ext x <;> simp [Multiset.mem_toFinset]`  err=`applyExtTheorem only applies to equations, not
  s.toFinset = {a} ↔ Multiset.card s ≠ 0 ∧ s = Multiset.card s • {a}`
  - solved=False `aesop (add simp [Multiset.mem_toFinset, Finset.eq_singleton_iff_unique_mem])`  err=`aesop: error in norm simp: tactic 'simp' failed, nested error:
maximum recursion depth has been reached
use `set_option maxRecDepth <num>` to increase limit
use `set_option diagnostics true` to get di`
  - solved=False `simp`
  - solved=False `simp_all`
  - solved=False `aesop`  err=`aesop: error in norm simp: tactic 'simp' failed, nested error:
maximum recursion depth has been reached
use `set_option maxRecDepth <num>` to increase limit
use `set_option diagnostics true` to get di`
  - solved=False `classical; aesop`  err=`<stdin>:1:9: expected '{' or tactic`
  - solved=False `ext x <;> simp`  err=`applyExtTheorem only applies to equations, not
  s.toFinset = {a} ↔ Multiset.card s ≠ 0 ∧ s = Multiset.card s • {a}`
- `Multiset.disjoint_toFinset` (rc1_solved=True): solved_probe=constructor <;> intro h <;> simp_all [Finset.disjoint_left, Multiset.mem_toFinset, Multiset.disjoint_left]; #solving_outcomes=1; run_error=None
  - minimality (unconfirmed, requires NS23 relabel): [{'probe': 'simp', 'solved': False}, {'probe': 'simp_all', 'solved': False}, {'probe': 'aesop', 'solved': False}, {'probe': 'ext x <;> simp', 'solved': False}]
  - solved=False `simp [Multiset.disjoint_toFinset]`  err=`kernel type check failed: (kernel) declaration has free variables '[anonymous]'`
  - solved=False `simp [Disjoint, Multiset.disjoint_left, Multiset.mem_toFinset]`
  - solved=False `simp only [Finset.disjoint_left, Multiset.mem_toFinset]`
  - solved=False `rw [Finset.disjoint_left]; simp [Multiset.mem_toFinset]`  err=`<stdin>:1:25: expected end of input`
  - solved=True `constructor <;> intro h <;> simp_all [Finset.disjoint_left, Multiset.mem_toFinset, Multiset.disjoint_left]`
- NS23 minimal-sufficient relabel is required before any solve counts as a win.

## 7. Candidate missing lemmas

| template | utility | novelty | risk |
|---|---|---|---|
| `multiset_mem_toFinset_bridge` | multi_theorem | probably_existing | duplicate |
| `multiset_disjoint_toFinset_bridge` | single_theorem | probably_existing | duplicate |
| `multiset_toFinset_eq_singleton_bridge` | single_theorem | probably_existing | duplicate |
| `multiset_mem_nsmul_bridge` | single_theorem | probably_existing | duplicate |
| `multiset_toFinset_ext_tactic_family` | multi_theorem | nearby_existing | promising |

Honest read: the bridge lemmas are **probably already in Mathlib**; the genuinely useful, `promising` item is the **`multiset_toFinset_ext_tactic_family`** (a narrow ext+membership probe), not a new lemma. The singleton failure is the first test case for it.

## 8. Recommendation

- **Do not modify RC1.**
- **Fix the SF1 measurement bug first**: `scripts/sf1_eval_matrix.py:parse_metrics` must read the `finished` key (not `proof_finished`/`solved`) so RC1 pass/fail is accurate; the SF1 frontier currently undercounts RC1 wins. (Non-protected file.)
- For the one genuine failure, the live probe ladder + NS23 minimality confirmation is the next step; if a singleton/membership-ext probe closes it where the induction oracle fails, it becomes the first **SF3 Lemma Inventor / tactic-family** candidate — a narrow gated action for `Multiset.toFinset` membership/singleton goals, explicitly *not* routed through the WX3 induction oracle.

## 9. Protected files

`git diff --stat HEAD` for protected configs (empty = unchanged):

```
(no changes to rc1_production_wrapper.json or ns24_router.json)
```

Working-tree status:

```
M README.md
?? project/evolve/experiments/sf1/
?? project/evolve/experiments/sf2/
?? project/evolve/reports/sf1_design.md
?? project/evolve/reports/sf1_live_eval_unblocker_status.md
?? project/evolve/reports/sf1_promotion_report.md
?? project/evolve/reports/sf1_stage_ab_status.md
?? project/evolve/reports/sf1_stage_cdef_status.md
?? project/evolve/reports/sf2_multiset_seed_report.md
?? scripts/sf1_backfill_frontier_paths.py
?? scripts/sf1_classify_frontier.py
?? scripts/sf1_common.py
?? scripts/sf1_eval_matrix.py
?? scripts/sf1_extract_mathlib_catalog.py
?? scripts/sf1_filter_consumed_surfaces.py
?? scripts/sf1_make_batches.py
?? scripts/sf1_minimal_relabel_new_wins.py
?? scripts/sf1_promotion_report.py
?? scripts/sf1_run_eval.py
?? scripts/sf2_build_failure_cases.py
?? scripts/sf2_extract_source_context.py
?? scripts/sf2_run_probe_ladder.py
```

No commit made. All SF2 changes are additive under scripts/sf2_*.py and project/evolve/experiments/sf2/ + this report.
