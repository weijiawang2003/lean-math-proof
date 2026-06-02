# FLI1 Part 0 — State Reconciliation

_Read-only inspection of FLI0 inputs and repo state. No FLI0 output altered._

## Repo state

- **HEAD:** `abd817e` ("Compare validated RC4B and RC4C candidates"), branch
  `tr5-ranker-guided-live-search`.
- **Dirty/untracked:** the in-progress RC5V3 raw run (modified `scripts/rc5v3_*.py`,
  `rc5_v3/out/*` logs/traces) + the committed-but-unstaged FLI0 tree. None of it is FLI1's; FLI1
  does not touch it.

## FLI0 seed input (verified, unmodified)

- `project/evolve/experiments/fli0/cases/fli0_seed_cases.json` — **present, 40 seeds**.
- **Namespace:** Finset 14, List 14, Multiset 4, Set 4, Nat 4.
- **Pattern:** SUBSET_BRIDGE 8, MAP_FILTER_BIND_BRIDGE 8, MEMBERSHIP_BRIDGE 7,
  INDUCTION_GENERALIZATION 6, SINGLETON_CHARACTERIZATION 4, IFF_SPLIT 4, DISJOINT_BRIDGE 2,
  EXTENSIONALITY_NEEDED 1.
- **Source stages:** RC5V2 21, RC5V3 19.
- Every seed carries `residual_goal_status = MISSING` and `residual_goal = null`.

## Residual goals missing → live rerun required

Confirmed: FLI0 artifacts (and the RC5V2/RC5V3 dynamic logs they derive from) record tactic
**outcomes** (`{rank, tactic, outcome}`), never the post-tactic **goal state**. So FLI1 must open
each seed live in LeanDojo to capture residual goals. **This is verified feasible** — a smoke test
on `Finset.biUnion_nonempty` opened a Dojo in ~3.7s and captured the residual goal after
`constructor` (two directional sub-goals `mp`/`mpr`) via `TransitionOutcome.next_state.pp`.

## Capability check for later parts

- **Live capture (Part 3):** LeanDojo `env.run_transition` → `next_state.pp` exposes residual
  goals. Reuses the `rc4b_gate.run_tactics_live` harness pattern (Dojo + SIGALRM per-tactic bound).
- **Typecheck / prove standalone lemmas (Parts 7–8):** `lake env lean` on a temp file importing
  the seed's source module typechecks a `:= by sorry` lemma in ~0.65s against the compiled Mathlib
  oleans at `~/.cache/lean_dojo/.../mathlib4/.lake/build/lib`. Fast and self-contained.
- **Faithful downstream rescue (Part 9):** these seeds are **real Mathlib lemmas** the *restricted*
  RC5 tactic battery failed to rediscover at the theorem's file position. A fresh `import Module`
  would put the theorem itself (and downstream) in scope, making rescue trivial and meaningless.
  Therefore the rescue test runs **in LeanDojo at the theorem's position** (where the lemma and
  later results are out of scope), using `have`-inlining of the candidate or `simp [existing_L]`,
  with controls that must fail. This is the honest test and the key FLI1 metric.

## Decision

Proceed with FLI1. Live rerun is **required and feasible**. The honest target is downstream rescue
through intermediate lemmas, not solved-count; direct solves during rerun are recorded as
`solved_directly` and explicitly excluded from FLI1 success.
