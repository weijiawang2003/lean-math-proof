# FLI2 Part 0 — State Reconciliation

_Read-only inspection of FLI0/FLI1 inputs. No FLI0/FLI1 output altered._

## Repo state

- **HEAD:** `009cec1` ("Extract residual goals and synthesize candidate lemmas for FLI1"),
  branch `tr5-ranker-guided-live-search`. FLI0 and FLI1 are committed.
- **Dirty/untracked:** the in-progress RC5V3 raw run (modified `scripts/rc5v3_*.py`,
  `rc5_v3/out/*`). Not FLI2's; untouched.

## FLI0 / FLI1 artifacts (verified present, unmodified)

| artifact | status |
|---|---|
| `fli0/cases/fli0_failed_cases_enriched.jsonl` | 455 cases, 327 clean, **file_path 455/455**, `top_retrieved_lemmas_detailed` present |
| `fli0/cases/fli0_failure_patterns.jsonl` | 455 |
| `fli0/cases/fli0_seed_cases.json` | 40 |
| `fli1/cases/fli1_candidate_lemmas_checked.jsonl` | 40 (21 PROBABLY_NEW, **15 EXISTS_CLOSE = 15 RETRIEVAL_GAP**, 4 too-vague) |
| `fli1/cases/fli1_candidate_lemmas_typechecked.jsonl` | 22 typecheck |
| `fli1/cases/fli1_candidate_lemma_proofs.jsonl` | 1 proved |
| `fli1/cases/fli1_downstream_rescue_results.jsonl` | **1 DOWNSTREAM_RESCUE**, 1 direct-dup, 38 no-rescue |
| `fli1/.../report` | present |

## Key counts for FLI2

- **FLI1 RETRIEVAL_GAP cases: 15** (all EXISTS_CLOSE flagged retrieval_gap).
- **FLI1 downstream rescues: 1** — `Finset.card_le_one_iff` via `simp [Finset.card_le_one] <;> aesop`
  (present and verified; all controls failed at position).
- **Broader FLI0 high-signal pool available: 217** clean failures with nonempty retrieved lemmas,
  namespace ∈ {Finset, List, Multiset, Set, Nat}, and a high-value bridge pattern
  (MAP_FILTER_BIND 101, SUBSET 39, IFF 36, MEMBERSHIP 20, INDUCTION 11, SINGLETON 6, DISJOINT 3,
  EXT 1). Far above the 100–250 target.
- **Live LeanDojo available:** yes (FLI1 used it; Dojo opens ~3.7s, `next_state.pp` exposes goals).

## Vacuity guard (carried from FLI1)

These are real Mathlib lemmas; their failure is position-dependent. FLI2 deploys retrieved lemma
`L` as `simp [L]`-style actions **at the theorem's LeanDojo position** (where the target theorem
and everything after it are out of scope), with controls that must fail. We never test against a
fresh `import Module` that would put the target theorem itself in scope. A solve where `L` is the
target theorem (or an alias of it) is flagged SELF_IMPORT_VACUOUS.

## Decision

Proceed with FLI2. Inputs are complete; live eval is available; pool size is ample. The research
question — can failure analysis identify reusable lemma-*deployment* rules that rescue downstream
theorems — is testable at scale here.
