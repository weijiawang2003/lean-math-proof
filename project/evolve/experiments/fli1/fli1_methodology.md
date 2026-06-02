# FLI1 Methodology

## Pipeline overview

`seeds → rerun-plan → live capture → cluster → synthesize → existing-check → typecheck → prove →
downstream-rescue → atlas/report`. Two live engines:

- **LeanDojo** (Parts 3, 9) — opens each real theorem at its file position; exposes residual goal
  states via `env.run_transition(...).next_state.pp`. Reuses the `rc4b_gate.run_tactics_live`
  harness (Dojo + per-tactic SIGALRM, process isolation via a driver/worker subprocess with hard
  `run_with_timeout` kill + per-seed checkpoint/resume).
- **`lake env lean`** (Parts 7, 8) — typechecks/proves standalone candidate lemmas in a temp file
  importing the seed's source module against the compiled Mathlib oleans (~0.65s/lemma). Temp
  files only; Mathlib source never touched.

## Part 2 — controlled rerun plan

Each seed gets a small, deterministic, pattern-specific tactic list designed to make *one* step of
progress and stop, so we capture a meaningful residual goal (not a giant search). Tactics:

- baseline probes: `simp`, the original failed dynamic tactic (if recorded).
- pattern openers: IFF/MEMBERSHIP/SINGLETON → `constructor`; SUBSET → `intro` / `rw [subset_iff]`
  shape; MAP_FILTER_BIND → `simp [retrieved_lemma]`; EXT → `ext x`; INDUCTION → one shallow
  `induction`/`cases` (timeout-bounded, one only).

Banned (per spec): `simp_all`, depth-3 try-chains, bare broad `aesop` loops, B20-style exhaustion.
Order-heavy seeds are low priority (none in the seed set — Order was excluded in FLI0). Plan sorted
by (priority, namespace, seed_id) for determinism.

## Part 3 — residual goal capture semantics

For each seed we open the Dojo, record `initial_goal = state0.pp`, then walk the plan tactics **in
sequence** (chaining `next_state`), recording each `next_state.pp`. Status:

- `solved_directly` — some tactic finished the proof. Recorded as proof-search evidence, **not**
  FLI1 success (no lemma invented).
- `captured` — at least one non-finishing, non-error tactic produced a residual goal.
- `timeout` / `infra_error` / `unknown_name` / `no_goal` / `needs_review` otherwise.

`capture_quality` = high if a clean post-opener residual goal was captured, medium if only the raw
initial goal, low/missing otherwise. Target: ≥25/40 captured.

## Part 4 — normalization & clustering

Normalize each residual goal pp: drop the hypothesis block, keep the `⊢` goal; α-rename locals
(`x y z…`), abstract type vars (`Type _`), strip inaccessible daggers (`inst✝`, `✝`) and universe
suffixes (`u_1`); tokenize constants; extract relation symbols
(`∈ ⊆ = ↔ ∀ ∃ Disjoint card map filter bind image singleton`). Cluster by
(namespace, pattern, main-relation, container-op). Conservative — unrelated goals stay separate.

## Part 5 — candidate lemma synthesis

For each high-confidence cluster, propose a small, local candidate lemma — preferably an `iff` or
implication that simplifies the residual goal. The Lean statement is built from the captured goal
(binders from the hyp context, inaccessible names sanitized), with the source module as the import.
Each candidate is tied to concrete `downstream_targets` (the seed theorems). We do **not** invent
large/vague statements; candidates with unresolved types are flagged for the typecheck stage.

## Part 6 — existing-lemma check

Classify each candidate `EXISTS_EXACT / EXISTS_CLOSE / PROBABLY_NEW / TOO_VAGUE_TO_CHECK /
ILL_TYPED_STATEMENT / NEEDS_REVIEW` using name search, statement-token & constant overlap, and the
FLI0 retrieved-lemma lists. **If a close lemma already exists but retrieval missed it → flag
`RETRIEVAL_GAP`** (itself a valuable discovery — the bridge exists, the searcher didn't use it).

## Parts 7–8 — typecheck & prove

Typecheck: `import <module>` + `lemma … := by sorry`; TYPECHECKS iff the only diagnostic is the
`sorry` warning. Errors classified (UNKNOWN_CONSTANT / TYPE_ERROR / MISSING_IMPORT / BINDER_ERROR /
UNIVERSE_OR_TYPECLASS_ERROR). Prove (only TYPECHECKS + conf≥medium + risk≤medium): safe tactics
(`simp`, `simp [L]`, `constructor <;> intro <;> simp at *`, `ext x <;> simp`, `simp [..] <;>
aesop`, narrow `omega`, one shallow induction). PROVED iff closes with no sorry/error. A proved
candidate is **not** a project success until it rescues downstream.

## Part 9 — downstream rescue (the real metric)

For each PROVED (or existing-close) candidate, at the original theorem's **LeanDojo position**:

1. Controls first — run the rescue tactic family **without** the candidate (e.g. `simp`,
   `constructor <;> simp`). If a control closes it → `DIRECT_SOLVE_DUPLICATE` (no credit).
2. Then with the candidate: inline `have <name> : <stmt> := by <proof>` (new lemmas) or reference
   the existing earlier lemma; apply `simp [name]` / `constructor <;> simp [name]` /
   `ext x <;> simp [name]`.
3. Classify `DOWNSTREAM_RESCUE` (closes, control failed) / `PARTIAL_PROGRESS` (goal simplified) /
   `NO_RESCUE` / `DIRECT_SOLVE_DUPLICATE` / `NEEDS_REVIEW`. Robustness = re-run twice.

≥1 true DOWNSTREAM_RESCUE is a milestone; PARTIAL_PROGRESS and RETRIEVAL_GAP findings are still
valuable.

## Determinism & safety

All non-live steps are pure functions of artifacts (sorted keys, no RNG/clock). Live steps are
checkpointed and resumable; capture order is fixed. No protected file is touched; temp Lean files
are deleted; no commit.
