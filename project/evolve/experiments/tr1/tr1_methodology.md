# TR1 — Failure-to-Action Router: methodology

## What TR1 is (and is not)

TR1 trains a **failure-to-action router/ranker**, *not* a proof generator. Given a theorem / failure
state, it predicts **which action family is worth trying next** — or that the case is
**no-cheap-action / missing-lemma / depth-gap**. It is a triage model over *verified proof-search
outcomes*, used to prioritize future work (SF5 retrieval, deeper search). It is **not** wired into
production routing and is **not** a theorem prover.

## Input (per example)

- theorem metadata: `full_name`, `file_path`, `namespace`
- theorem-name tokens (split on `.` `_` camelCase)
- goal / failure text where available (from live probe traces)
- trace symptoms (aesop-failed, simp-failed, max-recursion, parse-error, …)
- source cluster (SF4 cluster id), goal shape, tactic symptom
- previous probe outcomes (which controls/probes were tried and their result)

## Output (label)

A single action-family / triage label (see label set below).

## Verified-label discipline (the core rule)

**Only verified labels are used.** Positive labels require one of:

- an **actual literal-production delta** (RC1/RC2 credited component win), or
- a **minimal-relabel-confirmed** win (NS23-style / SX4 TRUE attribution), or
- a **historically accepted RC1/RC2 component** win.

**Negative / triage labels** come from:

- SF4 confirmed RC2 failures with no probe win (`NO_CHEAP_ACTION`),
- speculative gates / families with 0 true wins (`SOURCE_SPECIFIC_OR_REJECTED`),
- SX3 over-credit cases **reclassified** by SX4 (`SX3_PRODUCTION_SUBSUMED`),
- SF4 missing-lemma triage (`MISSING_BRIDGE_LEMMA_CANDIDATE`, `PROOF_SEARCH_DEPTH_GAP`),
- baseline-closed cases (`BASELINE_DUPLICATE`).

**Never train on unverified proxy wins as positive labels.** The SX3 depth-2 "wins" are included only
as the *negative* class `SX3_PRODUCTION_SUBSUMED` — they are the canonical example of what over-credit
looks like, and the router must learn *not* to chase them.

## Label set

| label | type | meaning |
|---|---|---|
| `SET_ITE_SIMP` | positive | single-shot `simp [Set.ite]` win (RC2 credited +5) |
| `WX3_MULTISET_INDUCTION` | positive | `Multiset.induction_on <;> simp_all` win |
| `MX2_TOFINSET_AESOP` | positive | `Set.Finite`/`toFinset`-gated `aesop` win |
| `BASELINE_DUPLICATE` | negative | a bare control (`simp`/`simp_all`/`aesop`) already closes it |
| `SX3_PRODUCTION_SUBSUMED` | negative | proxy thought it a win; literal production already solves it |
| `SOURCE_SPECIFIC_OR_REJECTED` | negative | only a source-specific `rw`/rejected family would close it |
| `NO_CHEAP_ACTION` | triage | confirmed RC2 failure, no cheap action found |
| `MISSING_BRIDGE_LEMMA_CANDIDATE` | triage | repeated shape, no generic tactic → reusable-lemma direction |
| `PROOF_SEARCH_DEPTH_GAP` | triage | closable in isolation but production search/routing missed it |

Labels with little support are still included in the label map and **flagged low-support** — the
honest outcome of a small verified corpus is a *pilot*, not a production router.

## Leakage controls

- Grouped evaluation (by namespace / source artifact) so the model can't memorize a single surface.
- Name-only vs name+goal ablation in error analysis to check whether the theorem name trivially
  determines the label (a leakage smell for the Set.ite cluster).
- No promotion to production routing; outputs feed a *next-work queue* only.
