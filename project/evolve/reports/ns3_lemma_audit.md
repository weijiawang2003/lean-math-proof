# NS3 — lemma-name audit for the 7 remaining `nat_defs_medium` failures

This document audits the 7 theorems still unsolved by v5-27-w4-master + NS1
(31/38 on `nat_defs_medium`). For each, we record:

  - the theorem statement,
  - why v5-27 fails on it,
  - the official Mathlib proof (commit 29dcec07, cached locally),
  - candidate premise lemmas with availability status,
  - candidate single-line priority templates derived from the Mathlib proof,
  - a **verdict**: promising / blocked / out of scope.

Availability data comes from `scripts/check_lean_names.py` querying
``~/.cache/lean_dojo/leanprover-community-mathlib4-29dcec074de168ac2bf835a77ef68bbe069194c5/mathlib4``,
with sibling-suffix detection to filter LeanDojo's trace-scratch copies
(``Defsgly9co8r.lean`` etc.) and same-file scope checking (a lemma
declared at or after the target line is reported CIRCULAR — Mathlib has
it but LeanDojo's trace replay won't expose it).

## Per-theorem

### A. `Nat.dvd_iff_div_mul_eq` (target: `Mathlib/Data/Nat/Defs.lean:1399`)

**Statement.** `d ∣ n ↔ n / d * d = n`.

**Why v5-27 fails.** The iff slot has a `dvd` template that uses
`⟨_, h.symm⟩` for the reverse direction. The `_` placeholder fails to
synthesise the divisor witness, leaving the `dvd_iff_div_mul_eq` goal
unclosed. Generic `omega` cannot close the goal either — it involves `∣`
and `*`.

**Mathlib proof (line 1399):**

```lean
lemma dvd_iff_div_mul_eq (n d : ℕ) : d ∣ n ↔ n / d * d = n :=
  ⟨fun h => Nat.div_mul_cancel h, fun h => by rw [← h]; exact Nat.dvd_mul_left _ _⟩
```

**Premise availability:**

| premise | status | source |
|---|---|---|
| `Nat.div_mul_cancel` | AVAILABLE | `Init/Data/Nat/Dvd.lean:91` |
| `Nat.dvd_mul_left`   | AVAILABLE | `Init/Data/Nat/Dvd.lean:16` |

**Candidate template** (iff slot, mid-specificity — depends on
``{hyp_*}``-free):

```
exact ⟨fun h => Nat.div_mul_cancel h, fun h => by rw [← h]; exact Nat.dvd_mul_left _ _⟩
```

**Verdict.** PROMISING. This is the verbatim Mathlib proof; both premises
are in scope. Replaces v5-27's broken `⟨_, h.symm⟩` form.

---

### B. `Nat.eq_one_of_mul_eq_one_left` (target: `Defs.lean:448`)

**Statement.** `(H : m * n = 1) : n = 1`.

**Why v5-27 fails.** No template in the `eq` shape — the goal `n = 1`
classifies as `eq` and v5-27 has no `eq` entries in `priority_templates`.
Fallback layers (omega, simp_all) can't close `(m*n=1) → n=1` because the
problem is non-linear over ℕ.

**Mathlib proof (line 448):**

```lean
lemma eq_one_of_mul_eq_one_left (H : m * n = 1) : n = 1 :=
  eq_one_of_mul_eq_one_right (by rwa [Nat.mul_comm])
```

**Premise availability:**

| premise | status | source |
|---|---|---|
| `Nat.eq_one_of_mul_eq_one_right` | AVAILABLE | `Defs.lean:445` (3 lines before target) |
| `Nat.mul_comm`                    | AVAILABLE | `Init/Data/Nat/Basic.lean:217` |
| `mul_eq_one` (generic monoid)     | AVAILABLE | `Mathlib/Algebra/Group/Units.lean:628` |

**Candidate templates** (new `eq` slot, both specific):

```
exact Nat.eq_one_of_mul_eq_one_right (by rwa [Nat.mul_comm])
simp_all [mul_eq_one]
```

**Verdict.** PROMISING. The first form is verbatim Mathlib; the second is
a generic-monoid fallback that exploits the imported `mul_eq_one` iff.
Adding an `eq` slot to `priority_templates` is the small genome change
required.

---

### C. `Nat.add_mod_eq_ite` (target: `Defs.lean:1243`)

**Statement.** `(m + n) % k = if k ≤ m % k + n % k then m % k + n % k - k else m % k + n % k`.

**Why v5-27 fails.** Documented in
`v5_failure_deep_dive_add_mod_eq_ite.md`: needs asymmetric branches —
each side of the `split_ifs` requires a different closing tactic. v5's
`split_ifs <;> simp_all <;> omega` only does symmetric propagation.

**Mathlib proof (line 1243):**

```lean
lemma add_mod_eq_ite :
    (m + n) % k = if k ≤ m % k + n % k then m % k + n % k - k else m % k + n % k := by
  cases k
  · simp
  rw [Nat.add_mod]
  split_ifs with h
  · rw [Nat.mod_eq_sub_mod h, Nat.mod_eq_of_lt]
    exact (Nat.sub_lt_iff_lt_add h).mpr (Nat.add_lt_add (m.mod_lt (zero_lt_succ _))
      (n.mod_lt (zero_lt_succ _)))
  · exact Nat.mod_eq_of_lt (Nat.lt_of_not_ge h)
```

**Premise availability:**

| premise | status | source |
|---|---|---|
| `Nat.add_mod`           | AVAILABLE | `Init/Data/Nat/Lemmas.lean:572` |
| `Nat.mod_eq_of_lt`      | AVAILABLE | `Init/Data/Nat/Div.lean:131` |
| `Nat.mod_eq_sub_mod`    | AVAILABLE | `Init/Data/Nat/Div.lean:137` |
| `Nat.sub_lt_iff_lt_add` | AVAILABLE | `Defs.lean:367` |
| `Nat.mod_lt`            | AVAILABLE | `Init/Data/Nat/Div.lean:142` |
| `Nat.lt_of_not_ge`      | AVAILABLE | `Init/Data/Nat/Basic.lean:447` |

**Candidate templates** (any-shape; multi-step):

```
cases k <;> simp_all <;> (rw [Nat.add_mod]; split_ifs with h <;> simp [Nat.mod_eq_of_lt, Nat.lt_of_not_ge, h])
rw [Nat.add_mod]; split_ifs with h; · rw [Nat.mod_eq_sub_mod h, Nat.mod_eq_of_lt (Nat.sub_lt_iff_lt_add h |>.mpr (Nat.add_lt_add (Nat.mod_lt _ (by omega)) (Nat.mod_lt _ (by omega))))]; · exact Nat.mod_eq_of_lt (Nat.lt_of_not_ge h)
```

**Verdict.** BLOCKED at single-line template granularity. The asymmetric
branches require either Lean 4 multi-line `·` bullet syntax (which
priority_templates' single-line renderer doesn't reliably parse) or a v6
branch-skeleton type with separate slots per branch. Documented as the
canonical case for the v6 architectural change.

We will still test the multi-step single-line form, guarded by the
existing `theorem_tactic_denylist` to prevent any DojoCrash from
poisoning later attempts on this target.

---

### D. `Nat.div_le_div_right` (target: `Defs.lean:544`)

**Statement.** `(h : a ≤ b) : a / c ≤ b / c`.

**Why v5-27 fails.** Goal shape is `le` (a / c ≤ b / c) — v5-27 has no
`le` slot. Fallback `omega` can't handle `/`. The generic `gcongr` tactic
is not currently in the genome.

**Mathlib proof (line 544):**

```lean
@[gcongr]
protected lemma div_le_div_right (h : a ≤ b) : a / c ≤ b / c :=
  (c.eq_zero_or_pos.elim fun hc ↦ by simp [hc]) fun hc ↦
    (le_div_iff_mul_le' hc).2 <| Nat.le_trans (Nat.div_mul_le_self _ _) h
```

**Premise availability:**

| premise | status | source |
|---|---|---|
| `Nat.le_div_iff_mul_le'` | AVAILABLE | `Defs.lean:529` (BEFORE target) |
| `Nat.div_mul_le_self`    | AVAILABLE | `Init/Data/Nat/Div.lean:247` |
| `Nat.le_trans`           | AVAILABLE | `Init/Prelude.lean:1677` |
| `Nat.eq_zero_or_pos`     | AVAILABLE | `Init/Data/Nat/Basic.lean:350` |
| `Nat.pos_of_ne_zero`     | AVAILABLE | `Init/Data/Nat/Basic.lean:354` |

**Candidate templates** (new `le` slot):

```
gcongr
by_cases hc : c = 0 <;> simp [hc] <;> exact (Nat.le_div_iff_mul_le' (Nat.pos_of_ne_zero hc)).2 (Nat.le_trans (Nat.div_mul_le_self _ _) {hyp_le})
```

**Verdict.** PROMISING (gcongr) + worth trying the inline form. `gcongr`
is a single-token tactic registered in `Mathlib/Tactic/GCongr.lean` and
the target is itself annotated `@[gcongr]` — so calling `gcongr` should
trigger this exact lemma. (Note: it'll only work if `gcongr` doesn't
attempt to use the target lemma recursively — needs an eval to confirm.)

---

### E. `Nat.sqrt_lt` (target: `Defs.lean:1620`)

**Statement.** `sqrt m < n ↔ m < n * n`.

**Why v5-27 fails.** The previous "sqrt_lt'" hypothesis from the V5
README is incorrect: `Nat.sqrt_lt'` IS available at `Defs.lean:1623`,
but only *after* the target line, so trace replay can't see it.
`Nat.le_sqrt` IS available BEFORE the target. v5-27 has no template for
sqrt-shaped iff goals.

**Mathlib proof (line 1620):**

```lean
lemma sqrt_lt : sqrt m < n ↔ m < n * n := by simp only [← not_le, le_sqrt]
```

**Premise availability:**

| premise | status | source |
|---|---|---|
| `Nat.le_sqrt` | AVAILABLE | `Defs.lean:1612` (8 lines BEFORE target) |
| `Nat.not_le`  | AVAILABLE | `Init/Data/Nat/Basic.lean:457` |

**Candidate template** (iff slot, specific):

```
simp only [← Nat.not_le, Nat.le_sqrt]
```

**Verdict.** PROMISING. Verbatim Mathlib proof, both premises in scope.

---

### F. `Nat.pow_lt_pow_iff_left` (target: `Defs.lean:745`)

**Statement.** `(hn : n ≠ 0) : a ^ n < b ^ n ↔ a < b`.

**Why v5-27 fails.** v5-27's iff slot has
`rw [Nat.pow_lt_pow_iff_left {hyp_ne_zero}]` — which is self-referential
(the target is `Nat.pow_lt_pow_iff_left`). Trace replay can't see the
lemma when proving itself.

**Mathlib proof (line 745):**

```lean
protected lemma pow_lt_pow_iff_left (hn : n ≠ 0) : a ^ n < b ^ n ↔ a < b := by
  simp only [← Nat.not_le, Nat.pow_le_pow_iff_left hn]
```

**Premise availability:**

| premise | status | source |
|---|---|---|
| `Nat.pow_le_pow_iff_left` | AVAILABLE | `Defs.lean:740` (5 lines BEFORE target) |
| `Nat.not_le`              | AVAILABLE | `Init/Data/Nat/Basic.lean:457` |

**Candidate template** (iff slot, specific; uses `{hyp_ne_zero}`):

```
simp only [← Nat.not_le, Nat.pow_le_pow_iff_left {hyp_ne_zero}]
```

**Verdict.** PROMISING. Verbatim Mathlib proof. Replaces v5-27's
self-referential `rw [Nat.pow_lt_pow_iff_left ...]` entry.

---

### G. `Nat.AM_GM` (target: `Defs.lean:1514`)

**Statement.** `{a b : ℕ} → (4 * a * b ≤ (a + b) * (a + b))`.

**Why v5-27 fails.** Declared `private lemma` (file-local). Even though
the target name resolves, the Mathlib proof would chain
`Nat.add_le_add`, `Nat.mul_le_mul`, and an arithmetic case-bash that
v5's heuristics can't generate. `nlinarith` and `polyrith` would close
it instantly but neither is in our trace-replay tactic surface.

**Premise availability:**

| premise | status | source |
|---|---|---|
| `Nat.add_le_add` | AVAILABLE | `Init/Data/Nat/Basic.lean:518` |
| `Nat.mul_le_mul` | AVAILABLE | `Init/Data/Nat/Basic.lean:722` |

**Candidate templates:** none worth wiring at the single-line layer.

**Verdict.** BLOCKED — environment-limited. Same conclusion as the v5
report; this needs `nlinarith` or a hand-coded multi-step proof too long
for a priority template. Out of scope for NS3.

## Summary table

| target | verdict | notes |
|---|---|---|
| `Nat.dvd_iff_div_mul_eq`        | PROMISING | replace existing dvd template |
| `Nat.eq_one_of_mul_eq_one_left` | PROMISING | new `eq` slot |
| `Nat.add_mod_eq_ite`            | BLOCKED   | needs branch skeleton (v6) |
| `Nat.div_le_div_right`          | PROMISING | new `le` slot; try `gcongr` |
| `Nat.sqrt_lt`                   | PROMISING | new iff template using `Nat.le_sqrt` |
| `Nat.pow_lt_pow_iff_left`       | PROMISING | replace self-ref template |
| `Nat.AM_GM`                     | BLOCKED   | needs nlinarith (out of scope) |

Five candidates promising; two blocked. The next stage builds five
variants — one per promising target plus a combined master — and runs
them on `nat_defs_medium`.
