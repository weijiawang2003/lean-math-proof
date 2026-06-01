# RC4B — Disjoint-left Bridge Candidate

**Candidate:** `RC4B = RC2 ⊕ disjoint_left bridge`

This workspace validates a **narrow** RC4 candidate family built around the
`disjoint_left` bridge lemma, which TR6 discovered to be **namespace-parametric**:
`Set.disjoint_left` and `Multiset.disjoint_left` are the same rewrite shape in two
namespaces, and both turn an opaque `Disjoint a b` goal into a membership statement
that `simp` (and optionally `aesop`) can close.

## Bridge lemmas

- `Set.disjoint_left`
- `Multiset.disjoint_left`

## Candidate actions

| action | tactic | namespace | bridge lemma |
|---|---|---|---|
| `SET_DISJOINT_LEFT_SIMP` | `simp [Set.disjoint_left]` | Set | `Set.disjoint_left` |
| `SET_DISJOINT_LEFT_SIMP_AESOP` | `simp [Set.disjoint_left] <;> aesop` | Set | `Set.disjoint_left` |
| `MULTISET_DISJOINT_LEFT_SIMP` | `simp [Multiset.disjoint_left]` | Multiset | `Multiset.disjoint_left` |
| `MULTISET_DISJOINT_LEFT_SIMP_AESOP` | `simp [Multiset.disjoint_left] <;> aesop` | Multiset | `Multiset.disjoint_left` |

Each action is **gated** on `requires_namespace` (Set / Multiset) **and**
`requires_name_or_goal_contains` ∈ {`disjoint`, `Disjoint`}, with
`max_emissions_per_theorem = 1`. The gate cannot fire on Nat / List / Finset / Order
goals (namespace mismatch), nor on Set/Multiset goals that never mention disjointness.

## What this candidate is NOT

This is a **narrow bridge candidate**. It deliberately excludes:

- broad `simp` with *all* disjoint lemmas
- `Finset.disjoint_*` (NOT independently evidenced here — kept as a negative control)
- `simp [*, disjoint]` / global `@[simp]` additions
- arbitrary `d2_simp_aesop` with any retrieved lemma (validated separately as RC4C)

Only the two named `disjoint_left` rewrites (and their `<;> aesop` depth-2 forms) in the
Set and Multiset namespaces are in scope.

## Evidence provenance

- **TR3** (`tr3_attribution.json`): 3 credited Set `disjoint_left` wins (depth-1 + depth-2).
- **TR5** (`tr5_rc4b_rc4c_evidence.json`): the same 3 Set wins reproduced live at rank 1
  (`READY_FOR_RC4B_VALIDATION`).
- **TR6** (`tr6_rc4b_rc4c_fresh_evidence.json`): **8 fresh** wins —
  4 Multiset (`disjoint_add_left`, `disjoint_cons_left`, `singleton_disjoint`,
  `zero_disjoint`) + 4 Set (`disjoint_iUnion_left/right`, `disjoint_sUnion_left/right`) —
  surfacing the namespace-parametric bridge and yielding
  `READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT`.

## Validation methodology

Mirrors the RC4A `def_unfold_simp` validation exactly (external additive evaluator):

1. extract known/fresh wins (Part 2)
2. define narrow candidate policy + shared gate (Part 3)
3. build validation theorem sets — known wins, fresh Set/Multiset holdouts, disjoint and
   namespace negative controls, canonical smoke (Part 4)
4. literal RC2 baseline, reuse-first (Part 5)
5. additive candidate evaluator: `candidate_solved = RC2_solved OR gated bridge action
   closes single-shot` (Part 6)
6. minimal attribution with bare controls (Part 7)
7. off-gate + preservation scan (Part 8)
8. determinism check (Part 9)
9. optional schema-native wrapper smoke (Part 10)
10. decision report (`project/evolve/reports/rc4/`) (Part 11)

**Promotion is not automatic.** This candidate is `off-by-default` and is not released,
not composed into an RC4 stack, and does not alter production routing (NS24) or the
frozen RC1/RC2 wrappers.
