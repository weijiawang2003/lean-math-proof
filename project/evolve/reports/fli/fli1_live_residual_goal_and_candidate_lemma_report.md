# FLI1 — Live Residual Goal Capture and Candidate Lemma Synthesis

**Decision: `FLI1_DOWNSTREAM_RESCUE_FOUND`**

FLI1 is the first real step from proof automation toward verifier-guided discovery. It re-ran the
40 FLI0 seed failures live to capture residual goals, synthesized candidate intermediate lemmas,
checked existence, typechecked and proved them, and tested **downstream rescue** at each theorem's
file position. The headline: **1 robust downstream rescue** plus a strong, honest signal that the
dominant gap is *retrieval/deployment of lemmas that already exist*. No production change, no
commit.

## 1. Executive summary

- **Residual goals captured: 40/40** (19 high-quality post-opener residuals, 21 medium = initial
  goal only). **0 solved directly** during rerun.
- **21 goal clusters** (8 multi-member) by (namespace, pattern, relation, container-op).
- **40 candidate lemmas** synthesized. Existing-check: 21 PROBABLY_NEW, **15 EXISTS_CLOSE — all 15
  flagged RETRIEVAL_GAP** (a close lemma was in the seed's retrieval set, but the restricted search
  never deployed it), 4 too-vague.
- **22/40 candidate statements typecheck** (target ≥10 met). 18 fail on universe/typeclass/type
  errors from reconstructing binders out of pretty-printed goal context.
- **1 candidate proved** standalone with safe tactics (`gcongr` on the card-monotonicity bridge).
  The rest are true but need their *specific* Mathlib closer, which the safe generic battery
  (simp/aesop/ext) does not supply — exactly why the restricted search failed.
- **1 robust DOWNSTREAM_RESCUE**: `Finset.card_le_one_iff` closes with
  `simp [Finset.card_le_one] <;> aesop` at its file position, where all 5 controls (incl. bare
  `aesop`) fail. The bridge `Finset.card_le_one` was retrieved but never deployed → a genuine
  retrieval-gap rescue. Re-run confirmed (robust).
- 1 DIRECT_SOLVE_DUPLICATE (a control closes it at-position — RC5's failure didn't reproduce
  there), 38 NO_RESCUE.

## 2. Why FLI1 follows FLI0

FLI0 produced 40 high-signal seed failures but could not capture **residual goal states** (the RC5
logs record tactic outcomes, not goals). Lemma invention needs the actual stuck goal. FLI1 supplies
it live, then closes the loop: goal → candidate lemma → proof → does it rescue the original?

## 3. Input seed corpus

40 FLI0 seeds (verified unmodified): Finset 14, List 14, Multiset 4, Set 4, Nat 4; patterns
SUBSET 8, MAP_FILTER_BIND 8, MEMBERSHIP 7, INDUCTION 6, SINGLETON 4, IFF 4, DISJOINT 2, EXT 1;
RC5V2 21 / RC5V3 19. See `experiments/fli1/state_reconciliation.md`.

## 4. Live residual goal capture

LeanDojo at each theorem's position (`env.run_transition(...).next_state.pp`), controlled
pattern-specific openers (constructor / intro / ext / `simp [L]`), per-tactic SIGALRM + per-seed
process hard-timeout + checkpoint. **40/40 captured, 0 solved_directly, 0 infra failures** (the
RC5V3 network outage did not recur). 19 high-quality residuals expose a genuine sub-goal (e.g.
after `constructor; intro h`); 21 medium captures only reproduce the initial goal because no safe
opener made progress.

## 5. Residual goal clustering

21 clusters; normalization strips universe suffixes / inaccessible daggers, abstracts type vars,
extracts relation symbols + container ops. 8 clusters have >1 member (e.g. a Finset `card_le_one_*`
family and a List `map/filterMap` membership family).

## 6. Candidate lemma synthesis

One candidate per captured seed: the residual hypothesis context becomes binders, the `⊢` goal the
conclusion (so the candidate is "given this context, the residual holds" = the intermediate lemma
the search stalled on). Statements reuse real Mathlib type vocabulary and import the seed's source
module. 29 are high-confidence/low-risk bridge candidates.

## 7. Existing lemma check

21 PROBABLY_NEW, **15 EXISTS_CLOSE (all RETRIEVAL_GAP)**, 4 TOO_VAGUE. The retrieval-gap finding is
itself valuable: for 15 seeds the bridge lemma was *already retrieved* — the failure is
routing/deployment, not a missing theorem. This is the RC4B/RC4C lesson restated at the lemma
level.

## 8. Typechecking

`lake env lean` against compiled Mathlib oleans (~1s/lemma). **22 TYPECHECKS**, 8 TYPE_ERROR, 10
UNIVERSE_OR_TYPECLASS_ERROR. The errors come from reconstructing binders out of pretty-printed
context (collapsed universes, dropped instance args) — a synthesis-quality limitation, not a
capture limitation (FLI2 fix below).

## 9. Candidate lemma proving

Safe tactics only (simp / `simp [L]` / constructor<;>simp / ext / `exact`·`apply`·`simpa using` the
close lemma / bounded `gcongr` for ⊆/≤ / omega). **1 PROVED** (`gcongr` on `s ⊆ t → s.card ≤ t.card`,
i.e. `Finset.card_le_card`). The other 21 typechecking candidates are true but need their specific
closer (e.g. `exact Finset.card_le_card h`) — the safe generic battery does not find it, which is
precisely the gap that made them RC5 failures. Honest result: standalone-proof yield is low because
the missing ingredient is a *specific* lemma/route, not raw search power.

## 10. Downstream rescue

Faithful test at each theorem's LeanDojo position (never a fresh full import, which would put the
theorem itself in scope). Controls (simp/aesop/constructor<;>simp/ext) run first; a candidate gets
credit only if a control fails and the candidate-deployment closes.

| outcome | count |
|---|---|
| **DOWNSTREAM_RESCUE (robust)** | **1** |
| DIRECT_SOLVE_DUPLICATE | 1 |
| NO_RESCUE | 38 |

## 11. Key example (the rescue)

- **Theorem:** `Finset.card_le_one_iff : s.card ≤ 1 ↔ ∀ {a b}, a ∈ s → b ∈ s → a = b`
- **Residual / pattern:** MEMBERSHIP/IFF cardinality characterization.
- **Candidate / bridge lemma:** `Finset.card_le_one` (`s.card ≤ 1 ↔ ∀ a ∈ s, ∀ b ∈ s, a = b`) —
  EXISTS_CLOSE, **retrieval gap** (it was retrieved for this seed but never deployed).
- **Controls at position (all fail to close):** `simp` ✗, `aesop` ✗ (progress only),
  `constructor <;> simp` ✗, `constructor <;> aesop` ✗, `ext x <;> simp` ✗.
- **Rescue (closes, robust):** `simp [Finset.card_le_one] <;> aesop`.
- **Reading:** the bridge already exists; the restricted RC5 battery just never applied it. A
  gated `simp [Finset.card_le_one]` enabling action (RC4B/RC4C deployment pattern) would have
  closed it. This is a genuine, reproducible downstream rescue through an intermediate lemma.

## 12. Limitations

- **Synthesis quality:** 18/40 candidate statements don't typecheck (pp-binder reconstruction).
  FLI2 should carry exact binder types from the goal's local context, not the pretty-print.
- **Standalone proof yield is low** under the safe battery; the real missing ingredient is a
  *specific* lemma/route. This is consistent with — not contradicting — the discovery thesis.
- **Rescue is mostly NO_RESCUE (38):** for non-retrieval-gap seeds we have no proved/known bridge to
  deploy, so there is nothing to test yet; these await better synthesis (FLI2).
- These seeds are real Mathlib lemmas, not open problems; "rescue" means the restricted searcher
  can now close them at-position, not that new mathematics was proved.

## 13. Recommended FLI2

1. **Fix candidate synthesis** to emit typechecking statements (exact binder types/universes;
   target ≥30/40 typecheck).
2. **Attack the 15 retrieval-gap bridges as a deployment problem**: build gated `simp [L]` enabling
   actions for the retrieved-but-unused lemmas (the RC4B/RC4C pattern) and measure at-position
   rescue across all 15 — this is the highest-yield, lowest-risk lever.
3. **Add a bounded specific-closer battery** (`gcongr`, `exact?`-style lemma application, `mono`)
   for PROBABLY_NEW bridges, then re-test rescue.
4. **Multi-step invention** for PARTIAL_PROGRESS: capture the new residual after a candidate and
   iterate.
5. Keep FLI an off-line discovery track; promote nothing into production.

## 14. Protected-file confirmation

`git diff --stat HEAD` over RC1/RC2/RC4-release/RC5S-policy wrappers + NS24 router = **empty**. No
RC*/TR*/FLI0 committed artifact, production wrapper, routing config, or README modified. FLI1 wrote
only under `project/evolve/experiments/fli1/`, `project/evolve/reports/fli/`, and
`scripts/fli1_*.py`, plus temp Lean files (deleted; Mathlib source untouched). Nothing promoted,
ranker not retrained, **no commit**.
