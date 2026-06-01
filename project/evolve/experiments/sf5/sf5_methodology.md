# SF5 — Existing-Lemma Retrieval for Missing-Bridge Targets

## Purpose

SF5 performs **retrieval before synthesis**. The SF4 failure-first miner and the
TR2 active-probing loop both converged on the same blocker: the current production
stack (RC2 = RC1 ⊕ SET_ITE_SIMP) has a residual pool of confirmed failures that no
cheap generic tactic or short sequence closes, and the largest cluster is the
*Set iff-equivalence* family. SF4 flagged 2 clusters as
`POSSIBLE_MISSING_BRIDGE_LEMMA` and TR2 attributed 20 cases as
`MISSING_BRIDGE_LEMMA_CANDIDATE`.

Before we attempt to *synthesize* any new lemma (deferred to a hypothetical SF6),
SF5 asks a cheaper question: **does Mathlib already contain a lemma that closes or
materially simplifies the goal, and is the gap merely that RC2's search/routing
never reaches it?** Inventing a lemma that already exists is the most expensive
possible mistake, so retrieval is mandatory first.

## Target definition

A theorem is a **missing-bridge target** if all of:

1. **literal RC2 failed** — confirmed through the production harness
   (`rc2_failure_confirmation.json`, classification `CONFIRMED_RC2_FAILURE`), not a
   custom runner.
2. **cheap probes failed** — SF4's four bare controls (`simp`, `simp_all`, `aesop`,
   `classical <;> aesop`), depth-1 sub-controls, and the SF4/TR2 short-sequence
   battery all failed on the theorem.
3. **cluster shape suggests an existing reusable lemma may close or simplify the
   goal** — the theorem sits in a cluster (chiefly Set iff-equivalence /
   Set ite-subset) whose members share goal shape, and TR2 labelled it
   `MISSING_BRIDGE_LEMMA_CANDIDATE`.

The target set is the deduplicated union of the TR2
`MISSING_BRIDGE_LEMMA_CANDIDATE` records and the SF4
`POSSIBLE_MISSING_BRIDGE_LEMMA` cluster members, intersected with confirmed literal
RC2 failures. Expected ≈ 20.

## Categories (attribution outcomes)

Every claimed win must be **over a confirmed literal RC2 failure** — never over a
custom depth-1 control battery (the SX3 over-credit lesson).

- **EXISTING_LEMMA_GAP** — Mathlib already has a lemma/theorem that solves or
  simplifies the target, and a retrieval-guided probe using that lemma closes the
  goal where literal RC2 failed. The proof is generic (an `exact`/`simpa`/`rw`/`simp`
  with the named lemma), not a copy of a long source-specific script.
- **RETRIEVAL_ROUTING_GAP** — the closing lemma exists *and* is reachable, but
  RC2's gated routing (e.g. aesop restricted to `Set.Finite/toFinset`) never tries
  it. The fix is a routing/retrieval tweak, not a new lemma.
- **TRUE_MISSING_BRIDGE_LEMMA** — no suitable existing lemma found after retrieval
  and live probing; the repeated cluster shape suggests a genuinely missing reusable
  bridge lemma. This is the only category that justifies SF6 synthesis.
- **PROOF_DEPTH_GAP** — existing lemmas are retrieved and help, but no single-step
  application closes the goal; multi-step proof planning is required.
- **NO_EVIDENCE / NO_RETRIEVAL_SIGNAL** — retrieval inconclusive: no lemma scored,
  or scored lemmas gave no live signal and no clear direction.

`BASELINE_DUPLICATE` (a trivial control already solves) and `PRODUCTION_SUBSUMED`
(literal RC2 actually solves / stale baseline) are guard classes: any target that
falls into them is removed from the missing-bridge accounting, mirroring the SX4
discipline.

## Method

1. **Lemma index** — scan the local LeanDojo-traced Mathlib source for the target
   files and nearby Set/Finset/Order source, plus the project declaration catalog
   (`discovered_theorems.json`), extracting declaration name + statement text where
   available. Where statement text is missing, fall back to name/token/path features.
   Report coverage; do not require perfect indexing.
2. **Retrieval** — for each target, rank lemma candidates by (a) lexical token
   overlap (TF-IDF / BM25-style), (b) namespace and file-path proximity, and (c)
   feature overlap over `{Set, iff, monotone, strictmono, subset, pairwiseDisjoint,
   compl, singleton, insert, ssubset, ite}`. Keep top-20 per target. Group targets by
   shared retrieved lemmas.
3. **Probes** — generate conservative, single-line probes per (target, lemma):
   `exact`, `simpa using`, `rw [..]` (iff/eq-shaped only), `simp [..]`, `simpa [..]`,
   `aesop (add simp [..])`. Plus cluster-level `simp only [l1,l2,..]` probes. Limits:
   ≤10 lemmas, ≤40 probes per target; malformed names skipped; parse risk recorded.
4. **Live probes** — open each theorem in LeanDojo, reconfirm RC2 failure (or reuse
   prior confirmation), run probes from the initial state under per-tactic SIGALRM
   and an OS hard timeout (driver/worker model, identical to SF4/TR2). Capture
   success / parse_error / proof_failed / timeout / not_found / open-flake.
5. **Attribution** — classify each target into the categories above; every win must
   beat literal RC2.
6. **Cluster analysis** — for the Set iff-equivalence cluster, determine whether one
   or two existing lemmas recur, whether retrieval can solve many targets, or whether
   each needs a separate theorem-specific lemma.
7. **Training export** — additive labelled examples for a future TR3/TR4
   retrieval-aware router. Never overwrite TR1/TR2 datasets.

## Guardrails

- Production configs (RC1/RC2 wrappers, NS24 router) are **read-only**; SF5 changes
  no production routing and creates no RC4.
- No lemma is synthesized in SF5.
- Source-specific long proof scripts are only emitted as `diagnostic` probes and are
  never counted as generic `EXISTING_LEMMA_GAP` wins.
