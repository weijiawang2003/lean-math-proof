# FLI2 Methodology

## Research question

Can failure analysis convert *retrieved-but-undeployed* lemmas into actionable, reusable
lemma-deployment rules that rescue failed downstream theorems? Success is measured by genuine
at-position downstream rescues and the deployment rules they support — not solved count.

## Pool (Part 2)

Three sources, deduped by theorem (FLI1 cases are a subset of the FLI0 seeds and carry richer
residual/rescue info, so they take precedence):

- **A. FLI1 RETRIEVAL_GAP** (15) — confirmed: a close lemma was retrieved but undeployed.
- **B. FLI1 EXISTS_CLOSE not fully rescue-tested** — exists-close candidates whose rescue was
  NO_RESCUE/untested.
- **C. FLI0 high-signal** (~217) — clean failures with nonempty retrieved lemmas, namespace ∈
  {Finset, List, Multiset, Set, Nat}, high-value bridge pattern, not RC2/RC4/RC5-solved, not
  unknown-name-only, not infra/timeout.

Prioritized patterns: MEMBERSHIP / SUBSET / MAP_FILTER_BIND / SINGLETON / DISJOINT / IFF / EXT /
INDUCTION. Deprioritized: ORDER_STRUCTURE_GAP, large algebraic goals, low-signal residuals,
unknown-name-only.

## Deployment actions (Part 3)

For each (theorem, retrieved lemma L) pair, generate small gated actions from the allowed templates
(`simp [L]`, `simp [L] <;> aesop`, `constructor <;> intro h <;> simp [L] at *`, `ext x <;> simp [L]`,
`intro x; simp [L] at *; aesop`, `exact L`/`exact L h`/`simpa using L`, `gcongr` for subset/card/
monotonicity, labeled `omega` only for Nat-light). Banned: simp_all, bare aesop as the credited
deployment, depth-3 chains, B20 search, unrestricted-namespace firing, unknown lemma names, long
induction. Each action is gated by namespace compatibility (theorem ns vs lemma ns), constant
overlap (theorem/goal ∩ lemma), pattern compatibility, and retrieved rank. Max 8 actions/theorem
(hard 12); only the top-ranked retrieved lemmas feed actions. Deterministic sort.

## Live eval (Parts 4–5) — vacuity-safe, at-position

The plan is **theorem-centric**: one LeanDojo Dojo per theorem (opened at its real file position,
so the target theorem and everything after are out of scope), running first the controls
(`simp`, `aesop`, `classical <;> aesop`, `constructor <;> simp`, `ext x <;> simp`) then every
candidate action for that theorem, all from the initial state, capturing `next_state.pp` for
residual-before/after. Process-group hard timeout + per-tactic SIGALRM + per-theorem checkpoint
(resume). We never use a fresh `import Module` (which would make the target theorem available and
the test vacuous). A candidate that solves while a control also solves → CONTROL_DUPLICATE. A
candidate that solves where `L` is the target theorem/alias → SELF_IMPORT_VACUOUS. Candidate wins
are re-run once for robustness.

## Attribution (Part 6)

TRUE_RETRIEVAL_GAP_RESCUE requires: RC5-source failure, lemma existed/retrieved, candidate-with-L
solves, controls fail, non-vacuous, robust. Other classes: PARTIAL_PROGRESS (goal strictly
simplified, fewer goals / shorter), CONTROL_DUPLICATE, BASELINE_DUPLICATE (RC2/RC4/RC5 actually
solves it / source mislabel), SELF_IMPORT_VACUOUS, UNKNOWN_NAME_OR_IMPORT_GAP (L not at position),
NO_RESCUE, NEEDS_REVIEW.

## Rule mining (Part 7)

Group TRUE rescues (and partials) by (namespace, pattern, lemma-family, template) into candidate
DEPLOYMENT_RULEs (FINSET_CARD_BRIDGE, FINSET_SUBSET_BRIDGE, LIST_MEM_MAP_FILTER_BRIDGE,
MULTISET_TOFINSET_BRIDGE, DISJOINT_MEMBERSHIP_BRIDGE, …). Each records trigger conditions, lemma
family, recommended actions, supporting rescues, partials, false-positives (actions that fired but
produced CONTROL_DUPLICATE/NO_RESCUE), risk, and promotion_status (candidate / needs_more_data /
reject). Discovery only — no rule is promoted into a wrapper.

## RC4B/RC4C comparison (Part 8)

RC4B/RC4C were *manually* validated lemma-enabling static wrappers. FLI2 tries to *discover*
analogous deployment actions from failure analysis. We report family overlap, new families beyond
RC4B/RC4C, and whether retrieval-gap deployment could become a future RC-candidate *generator*.

## Determinism & safety

Non-live steps are pure functions of artifacts (sorted, no RNG/clock). Live steps checkpoint and
resume; fixed action order. No protected file touched; no commit.
