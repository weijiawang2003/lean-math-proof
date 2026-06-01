# RC4C — d2_simp_aesop candidate

**Candidate:** `RC2 ⊕ narrow d2_simp_aesop`.

## Action shape

    simp [L] <;> aesop

where `L` is one of a **small allowlisted set** of retrieved lemmas, each validated by
TR3 / TR5 / TR6 evidence and each carrying its own narrow namespace + name/goal gate.
The mechanism is *depth-2*: a single named-lemma `simp [L]` rewrite that advances the
goal into a shape `aesop` can then close, where neither bare `aesop` nor `simp [L]`
*alone* closes it. (If `simp [L]` alone already closes the goal, that is RC4A / d1
territory — not a true RC4C member; the minimal attribution demotes it to
`SIMP_ONLY_DUPLICATE`.)

## Allowlisted lemmas (the entire candidate)

| lemma `L` | namespace | gate tokens | overlap |
|---|---|---|---|
| `Set.disjoint_left`        | Set      | disjoint/Disjoint | **RC4B** |
| `Multiset.disjoint_left`   | Multiset | disjoint/Disjoint | **RC4B** |
| `Multiset.disjoint_right`  | Multiset | disjoint/Disjoint | none |
| `Set.subset_pair_iff_eq`   | Set      | pair/subset_pair  | none |
| `Finset.biUnion_subset`    | Finset   | biUnion           | none |
| `List.forall_iff_forall_mem` | List   | Forall            | none |

`simp [<NS>.disjoint_left] <;> aesop` is **literally an RC4B action** (RC4B already
validated the depth-2 `<;> aesop` form of the disjoint_left bridge). Those two actions
are therefore tagged `overlap_family: RC4B` and are **excluded from the pure-RC4C
(non-overlap) policy**, which keeps only the four lemmas not already covered by RC4B.

## This candidate is NOT

- broad arbitrary depth-2 sequence search;
- all retrieved lemmas (only the 6 allowlisted, each with a specific gate);
- generic `simp <;> aesop` everywhere (every action requires a named lemma `L`);
- `simp_all <;> aesop` (not independently evidenced);
- the RC4B disjoint_left bridge silently re-counted — its two actions are tagged as
  overlap and reported separately so RC4C's *pure* (non-overlap) delta is honest.

## Evaluation modes

Two policies are evaluated separately:

- **RC4C_all** — all 6 allowlisted actions (includes the 2 RC4B-overlap actions).
- **RC4C_nonoverlap** — the 4 actions whose lemma is *not* `Set/Multiset.disjoint_left`
  (the genuinely new-to-RC4 material).

External additive evaluator: `candidate_solved = literal_RC2_solved OR (gate fires AND
some gated `simp [L] <;> aesop` closes the goal single-shot from the initial state)`,
so regressions are structurally impossible (candidate ⊇ RC2). Off-by-default; **not**
promoted; **no** RC4 release.

## Artifacts

`theorem_sets/` validation sets + manifest · `out/` literal-RC2 / candidate / attribution
/ off-gate / determinism / schema-smoke results · `d2_simp_aesop_policy.json` the policy ·
report under `project/evolve/reports/rc4/rc4c_d2_simp_aesop_validation_report.md`.
