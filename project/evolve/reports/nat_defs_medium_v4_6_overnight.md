# nat_defs_medium — v4.6 overnight controlled sweep

## Summary

| variant            | div templates | div budget | proved   | by_origin (fb/fam/gen)   | retrieval (att / adv / win) | shape_filter_drops |
|--------------------|---------------|------------|----------|--------------------------|------------------------------|---------------------|
| v45 (reference)    | 19            | 20         | 25 / 38  | 18 / 4 / 3               | 354 / 0 / 0                  | 238                |
| verified           | 15            | 20         | 25 / 38  | 18 / 4 / 3               | 354 / 0 / 0                  | 238                |
| **constructor**    | **6**         | **8**      | **26 / 38**  | **18 / 4 / 4**           | **262 / 7 / 0**              | **177**            |
| div-rewrite        | 8             | 10         | 25 / 38  | 18 / 4 / 3               | 354 / 0 / 0                  | 238                |
| mixed-small        | 11            | 14         | 25 / 38  | 18 / 4 / 3               | 354 / 0 / 0                  | 238                |
| verified-no-rw-eq  | 14            | 18         | 25 / 38  | 18 / 4 / 3               | 314 / 7 / 0                  | 213                |

`v45`, `verified`, `div-rewrite` and `mixed-small` produce **byte-identical**
proof-stage metrics. The verifier drops 4 templates from `verified`, but
none of them were the first to fire on any div theorem, so the proof
trajectory does not change.

`constructor` is the only variant that beats v4.5: **26/38, +1**.

`verified-no-rw-eq` is the hypothesis-confirmation variant — it drops
`rw [Nat.div_eq_of_lt]` from the v4.5 div family but leaves the rest
intact. It stays at 25/38, refuting the simpler hypothesis that
`rw [Nat.div_eq_of_lt]` alone was the derailing tactic.

## Mechanism of the +1 closure

The newly-proved theorem is **`Nat.div_lt_iff_lt_mul'`**.

### v4.5 trajectory (proof exhausted)

```
step 1  rw [Nat.div_eq_of_lt]              family_tactic   1→2  (stray side-goal)
step 2  rw [Nat.div_eq_of_lt]              family_tactic   2→err
        rw [Nat.div_lt_iff_lt_mul']        family_tactic   2→err   (pattern doesn't match the new state)
        rw [Nat.div_lt_iff_lt_mul]         family_tactic   2→err
        rw [Nat.div_le_iff_le_mul]         family_tactic   2→err
        rw [Nat.div_lt_iff_lt_mul hb,...]  family_tactic   2→err
        (retrieved-premise rewrites all fail; goal state was already corrupted by step 1)
        ...
        exhausted at step 8
```

### verified-no-rw-eq trajectory (proof exhausted, different derail)

```
step 1  rw [Nat.div_lt_iff_lt_mul']        family_tactic    1→1   (advances, but to a state simp_all can't close)
step 2  apply Nat.lt_of_lt_of_le           retrieved_apply  1→3   (bloat — skipped on subsequent attempts)
        apply Nat.pos_of_ne_zero           retrieved_apply  1→1
step 3  simp [Nat.zero_mul]                generative_topk  1→1
        ...
        exhausted at step 8
```

### constructor variant trajectory (closed!)

```
step 1  rw [Nat.div_lt_iff_lt_mul]         retrieved_premise  1→1   (no-prime form, cleaner rewrite target)
step 2  simp_all                           generative_topk    1→0   ProofFinished
```

### Why the no-prime variant wins

`Nat.div_lt_iff_lt_mul` (no prime) and `Nat.div_lt_iff_lt_mul'` (prime)
have different statements:

  - `Nat.div_lt_iff_lt_mul`:  `0 < k → (n / k < m ↔ n < m * k)`
  - `Nat.div_lt_iff_lt_mul'`: `0 < k → (n / k < m ↔ n < k * m)`

The goal of `Nat.div_lt_iff_lt_mul'` is itself the prime form. So:

  - Rewriting with the **prime** form (`Nat.div_lt_iff_lt_mul'`) leaves
    a state whose simp normal form drops the `0 < k` premise into a
    side-condition that `simp_all` doesn't dispatch.
  - Rewriting with the **no-prime** form (`Nat.div_lt_iff_lt_mul`)
    introduces a `Nat.mul_comm`-equivalent reordering that `simp_all`
    *can* normalize via its commutativity lemmas.

In v4.5 the family tries `rw [Nat.div_lt_iff_lt_mul']` *before*
retrieval can offer `rw [Nat.div_lt_iff_lt_mul]`. In the constructor
variant the family has no rewrite templates, so retrieval gets first
shot, and the static catalog ranks the no-prime form first (it lacks
the trailing apostrophe, scoring better on token overlap with the
goal's `n / k < m` head).

## Variant configurations

| variant            | family content                                                                                  |
|--------------------|-------------------------------------------------------------------------------------------------|
| v45                | full v4.5 list — `[omega, simp, simp_all, simp [div_eq_of_lt], simp [div_eq_of_lt, lt_of_lt_of_le], rw [div_eq_of_lt], rw [div_lt_iff_lt_mul'], rw [div_lt_iff_lt_mul], rw [div_le_iff_le_mul], exact div_le_div_right ‹_›, apply div_le_div_right, simp [div_lt_iff_lt_mul, mul_one], simp_all [div_lt_iff_lt_mul, mul_one], simp_all [div_lt_iff_lt_mul', mul_one], rw [div_lt_iff_lt_mul {hyp_pos}, mul_one], constructor <;> intro h_split <;> omega, constructor <;> intro h_split <;> simp_all, induction {hyp_le} <;> simp_all, induction {hyp_le} with | refl => exact le_refl _ | step h_step ih => exact ih.trans (div_le_succ_div _ _)]` |
| verified           | v45 minus the 4 templates that reference `Nat.div_le_div_right` / `Nat.div_le_iff_le_mul` / `Nat.le_refl` (filtered by `evolve.template_verifier`) |
| constructor        | `[omega, simp, simp_all, constructor <;> intro h <;> omega, constructor <;> intro h <;> simp_all, constructor <;> intro h <;> simp_all <;> omega]`              |
| div-rewrite        | `[omega, simp, simp_all, simp [div_eq_of_lt], simp [div_eq_of_lt, lt_of_lt_of_le], rw [div_eq_of_lt], rw [div_lt_iff_lt_mul'], rw [div_lt_iff_lt_mul]]`         |
| mixed-small        | constructor ∪ div-rewrite                                                                       |
| verified-no-rw-eq  | verified minus `rw [Nat.div_eq_of_lt]` only                                                     |

## Template verification audit

Verifier output for the **v45** default div family (19 templates):

| metric                                       | value                                                          |
|----------------------------------------------|----------------------------------------------------------------|
| `template_count`                             | 19                                                             |
| `template_constant_checked_count`            | 9                                                              |
| `template_constant_available_count`          | 6                                                              |
| `template_constant_unavailable_count`        | 2 (`Nat.div_le_div_right`, `Nat.div_le_iff_le_mul`)            |
| `template_constant_type_mismatch_count`      | 1 (`Nat.le_refl`)                                              |
| `filtered_template_count`                    | 4                                                              |
| `filtered_templates`                         | see below                                                      |

Templates filtered by `evolve/template_verifier.py`:

  1. `rw [Nat.div_le_iff_le_mul]`
  2. `exact Nat.div_le_div_right ‹_›`
  3. `apply Nat.div_le_div_right`
  4. `induction {hyp_le} with | refl => exact Nat.le_refl _ | step h_step ih => exact ih.trans (Nat.div_le_succ_div _ _)`

None of these ever advanced a div theorem in v4.1–v4.5; removing them
does not regress the proof count and trims 4 guaranteed-fail Lean
roundtrips per div-theorem state. Net diagnostic improvement.

## Per-theorem div/dvd status table (constructor variant)

| theorem                       | status   | num_steps | closing tactic                |
|-------------------------------|----------|-----------|-------------------------------|
| Nat.div_le_div_right          | EXH      | 8         | —                             |
| **Nat.div_lt_iff_lt_mul'**    | **PROVED** | **2**   | **`simp_all` after retrieval `rw [Nat.div_lt_iff_lt_mul]`** |
| Nat.div_lt_one_iff            | EXH      | 8         | —                             |
| Nat.div_pos                   | ERROR    | 5         | All top-40 tactics errored at step 5 |
| Nat.div_pos_iff               | ERROR    | 4         | All top-40 tactics errored at step 4 |
| Nat.dvd_iff_div_mul_eq        | ERROR    | 3         | All top-40 tactics errored at step 3 |

The five remaining div/dvd targets still fail. ERROR (not EXH) on the
last three indicates the rollout exhausted candidate tactics at the
current state, not the max-step budget — characteristic of states the
generative top-k + family + retrieval list cannot advance past.

## Answers to the spec questions

1. **Did any new theorem solve beyond 25/38?**
   Yes — **`Nat.div_lt_iff_lt_mul'`** in the constructor variant.
   First closure of a div theorem in this branch.

2. **Did `Nat.dvd_iff_div_mul_eq` move from EXHAUSTED to PROVED?**
   No. It moved from EXH to ERROR at step 3 in the constructor variant
   (and stayed ERROR in verified / verified-no-rw-eq). The earlier
   error indicates the family ordering change collapses the search
   tree faster, not that the proof was found.

3. **Did any div theorem close?**
   Yes — `Nat.div_lt_iff_lt_mul'`, exactly one.

4. **Which template came closest?**
   No template *closed* the proof — the closer was the generative
   `simp_all` from the underlying gen_v5 top-k. The template work
   merely opened the door by removing the family rewrite that derailed
   the state at step 1. Of the new v4.5 structured templates that
   v4.6 kept, none fired as the closing tactic on any theorem.

5. **Which templates should be disabled?**
   Filtered by the verifier (effective immediately):
     - `rw [Nat.div_le_iff_le_mul]`
     - `exact Nat.div_le_div_right ‹_›`
     - `apply Nat.div_le_div_right`
     - `induction {hyp_le} with | refl => exact Nat.le_refl _ | step ...`
   Recommended disable (no advance, no closure across v4.1–v4.5):
     - `rw [Nat.div_eq_of_lt]` — derails iff-shape div goals as shown.
     - `simp [Nat.div_lt_iff_lt_mul, Nat.mul_one]` — 0 advances, 39 errors.
     - `simp_all [Nat.div_lt_iff_lt_mul, Nat.mul_one]` — 0 advances.
     - `simp_all [Nat.div_lt_iff_lt_mul', Nat.mul_one]` — 0 advances.
     - `induction {hyp_le} <;> simp_all` — 0 advances.

6. **Is retrieval still useful, or is it only diagnostic?**
   It is now load-bearing for closures, not just diagnostic. In the
   constructor variant retrieval contributes step-1 advances on
   3 div theorems (262 attempts, 7 advances) and step-1 of the only
   newly-closed div proof. The shape filter still drops 177 mismatched
   forms per run. Without retrieval, removing `rw [Nat.div_eq_of_lt]`
   from the family would not have produced the +1 closure.

7. **What should v4.7 target?**
   - **Path A (recommended) — drop the demonstrably-useless v4.5
     templates from the v4.6 verified config and treat constructor as
     the new baseline.** Re-run with a div family of just `[omega,
     simp, simp_all] ∪ constructor templates`. Apply the same
     verification pass to all family lists (not just div).
   - **Path B — small evolutionary sweep.** With constructor as the
     new seed, run `--generations 2 --population-size 4 --survivors 2`
     and let the mutator try small permutations of div family order
     and budget. The +1 closure was found by manual ablation; a
     budget-constrained mutator should find similar wins.
   - **Path C — term-mode builder.** The three div theorems still
     erroring at step 3-5 (`Nat.div_pos`, `Nat.div_pos_iff`,
     `Nat.dvd_iff_div_mul_eq`) get stuck in states the generative
     model can't advance. A term-mode proof builder (build the
     proof term directly via `exact ⟨_, _⟩` synthesis) is the next
     research direction if tactic search continues to plateau.

## Runtime / wall-clock

| variant            | wall-clock |
|--------------------|------------|
| v45                | ~4 min 50s |
| verified           | ~4 min 55s |
| constructor        | ~4 min 09s |
| div-rewrite        | ~4 min 43s |
| mixed-small        | ~13 min (slower per-state retrieval scoring at higher family budget) |
| verified-no-rw-eq  | ~4 min 40s |

Total sweep wall-clock: **~36 minutes**.

## Artifacts

  - `project/evolve/runs/evolve-20260522-061049-9be813/`  — v45
  - `project/evolve/runs/evolve-20260522-061553-122264/`  — verified
  - `project/evolve/runs/evolve-20260522-062048-3673c6/`  — **constructor (winner)**
  - `project/evolve/runs/evolve-20260522-062457-6e9f3e/`  — div-rewrite
  - `project/evolve/runs/evolve-20260522-062940-b62ae9/`  — mixed-small
  - `project/evolve/runs/evolve-20260522-064654-332077/`  — verified-no-rw-eq
  - `evolve/template_verifier.py`                          — new module
  - `evolve/run_evolve.py`                                 — `--template-variant` CLI
  - `project/evolve/reports/v4_6_template_failure_diagnostics.md` — Stage 2 census
  - `project/evolve/reports/nat_defs_medium_v4_6_overnight.md`     — this file

## Recommendation

Adopt the **constructor** variant as the v4.6 seed (or merge it as the
new default for `theorem_family_tactics["div"]`). It strictly improves
the proved count (26/38 vs 25/38), uses fewer Lean roundtrips per
state, and produces cleaner diagnostics. The verifier infrastructure
(`evolve/template_verifier.py`) is safe to keep on by default for all
future variants — it cleanly drops templates whose constants are known
to be unavailable.

The five remaining div failures are not template-tractable on the
current generative checkpoint. The next research direction is either
to retrain gen_v5 with div-family-augmented data, or to introduce a
term-mode proof builder for the `Nat.dvd_iff_div_mul_eq` / `Nat.div_pos*`
shape.
