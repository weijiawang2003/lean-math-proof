# RC4D — RC4 Composition Candidate

RC4D is the **first composed RC4 candidate**. It is **not a release** and **not promoted**.
It exists to test whether the three independently-validated RC4 components stack additively
over RC2 without double-counting, off-gate firing, regressions, or floor loss — the gate
that any eventual RC4 release candidate must pass first.

    RC4D = RC2 ⊕ RC4A ⊕ RC4B ⊕ RC4C_residue

## Components

| Component | Source | Actions kept | Mechanism |
|---|---|---|---|
| **RC4A** def_unfold_simp | `def_unfold_simp/` (RC4A_CANDIDATE_CONFIRMED) | `simp [<allowlisted def>]` depth-1 over the 9-def allowlist (Monotone/MonotoneOn/Antitone/AntitoneOn/StrictMono/StrictMonoOn/StrictAnti/StrictAntiOn/Finset.disjUnion) | goal-driven definitional unfold |
| **RC4B** disjoint_left_bridge | `disjoint_left_bridge/` (RC4B_CANDIDATE_CONFIRMED) | `simp [<NS>.disjoint_left]` + `<;> aesop` for NS ∈ {Set, Multiset} on disjoint goals | namespace-parametric disjoint bridge |
| **RC4C_residue** | `d2_simp_aesop/` (RC4C_CONFIRMED_WITH_RC4B_OVERLAP) | **non-overlap residue only** | depth-2 `simp [L] <;> aesop` deployed RC4B-style |

## RC4C_residue (de-duplicated)

RC4C was confirmed only *with RC4B overlap*: 8 of its 12 evidence wins use
`Set/Multiset.disjoint_left` — literally RC4B actions. The composition must not re-credit them.
RC4C_residue therefore keeps **only the non-overlap allowlist lemmas** that RC4C minimal
attribution credited as genuine depth-2 (`simp [L]` alone fails):

- `Multiset.disjoint_right`  (MULTISET_DISJOINT_RIGHT_D2)
- `Set.subset_pair_iff_eq`   (SET_SUBSET_PAIR_D2)
- `List.forall_iff_forall_mem` (LIST_FORALL_D2)

Explicitly **excluded** from RC4C_residue:

- `Set.disjoint_left`, `Multiset.disjoint_left` — overlap RC4B (dropped, `drop_rc4c_overlap_rc4b`).
- `Finset.biUnion_subset` — **SIMP_ONLY_DUPLICATE**: `simp [Finset.biUnion_subset]` closes its
  evidence theorem *alone* (depth-1, not depth-2). Dropped from depth-2 credit
  (`drop_simp_only_duplicate_depth2_credit`).

### Theorem-level overlap (the de-dup that matters)

Several RC4C "pure" wins are on Multiset disjoint theorems that **RC4B already solves** via
`disjoint_left` (e.g. `Multiset.disjoint_add_left/_right`, `disjoint_iff_ne`, `disjoint_union_left`,
`singleton_disjoint`). With ordering `[RC4A, RC4B, RC4C_residue]`, those theorems are credited
to **RC4B** (earlier component), so RC4C_residue's genuinely-additive *theorem* coverage is the
small set no earlier component closes (e.g. `List.Forall.imp`, `Set.Nonempty.subset_pair_iff_eq`).
RC4C_residue's `Multiset.disjoint_right` action is retained because it is a *distinct mechanism*,
but its credit accrues only where RC4B does not already win.

## Deployable form

RC4C's original fused `simp [L] <;> aesop` failed schema-native smoke (0/12): the best-first
search applies the `<;>` combinator differently than a single-shot transition. RC4B got 10/11
because it *also* prepended the **bare `simp [L]`** enabling action, letting the search's own
aesop close the advanced state. RC4D therefore deploys every RC4B/RC4C_residue lemma RC4B-style:
**both `simp [L]` and `simp [L] <;> aesop`**, name-gated.

## Evaluation modes

- **External additive** (authority): `candidate_solved = literal_RC2 OR (a gated component
  tactic closes single-shot)`; ordered component attribution de-duplicates credit. Additive ⇒
  regressions structurally impossible.
- **Schema-native wrapper** (deployability): RC2 copy + gated component tactics prepended to
  `priority_templates["any"]`; smoke-tested through the real `eval_rollout_all` search.

## Gate / validation status

RC4D must pass composition validation (positive credited delta, schema wrapper reproduces most
additive wins, 0 off-gate, 0 regressions, deterministic, canonical floors preserved, RC4C residue
de-duplicated) **before** any RC4 release candidate is considered. Off-by-default; protected RC1/
RC2/NS24 + RC4A/B/C source artifacts untouched; no commit.
