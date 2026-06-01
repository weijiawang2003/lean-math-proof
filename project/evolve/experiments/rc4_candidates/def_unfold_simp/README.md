# RC4A candidate — `def_unfold_simp`

**Candidate:** RC4A = RC2 ⊕ `def_unfold_simp` (experimental; **not promoted**).

This candidate is the narrowest of the three TR3 `FOUND_RC_CANDIDATE_FAMILY`
families. It is built from the 5 TR3 `TRUE_RETRIEVAL_ONLY_DELTA` wins whose winning
mechanism was a **definitional unfold** — `simp [D, …]` where `D` is a Mathlib
*definition* (not `@[simp]`) whose name appears in the goal:

| TR3 win | winning tactic |
|---|---|
| `Set.monotoneOn_iff_monotone` | `simp [Monotone, MonotoneOn]` |
| `Set.antitoneOn_iff_antitone` | `simp [Antitone, AntitoneOn]` |
| `Set.strictMonoOn_iff_strictMono` | `simp [StrictMono, StrictMonoOn]` |
| `Set.strictAntiOn_iff_strictAnti` | `simp [StrictAnti, StrictAntiOn]` |
| `Finset.mem_disjUnion` | `simp [Finset.disjUnion]` |

RC2 fails on these only because the definitions are not `@[simp]`, so RC2's bare
`simp`/`aesop` never unfolds them.

## What this candidate IS

A **goal-driven definitional unfold restricted to a validated allowlist** of exactly
the definitions that produced TR3 wins
(`Monotone/MonotoneOn`, `Antitone/AntitoneOn`, `StrictMono/StrictMonoOn`,
`StrictAnti/StrictAntiOn`, `Finset.disjUnion`). For a theorem, the gate fires **only**
if one of these allowlisted definitions appears in the goal/statement, and then emits
a single `simp [<the allowlisted defs present in the goal>]`.

## What this candidate is NOT

- **Not** broad `simp` unfolding of arbitrary definitions.
- **Not** `simp [all retrieved lemmas]` / large `simp only` lists.
- **Not** adding global `@[simp]` lemmas.
- **Not** the `d2_simp_aesop` family.
- **Not** the `Set.disjoint_left` bridge (validated separately as RC4B if RC4A passes).

## Validation methodology (mirrors SET_ITE_SIMP → RC2)

External **additive** evaluator (avoids search-perturbation artifacts):
`candidate_solved = literal_RC2_solved OR (gate fires AND gated simp closes)`.
A new win is credited only when literal RC2 failed, the gated `simp [defs]` closes,
and minimal attribution shows bare controls (`simp`/`simp_all`/`aesop`/`classical <;>
aesop`) do **not** close it. Off-gate (firing on negative controls) and canonical
floors (demo_v1 / nat_defs_medium / nat_defs_large) must be clean; the gated eval must
be deterministic.

Promotion is **not** performed here — RC4A produces a decision only.
