# SF4 — RC2 Failure-First Frontier Miner: methodology

## The new rule

> **A candidate (tactic / sequence / lemma) is mined and credited only if it improves over the
> LITERAL RC2 production baseline.** Mining starts from **literal RC2 failures**, never from a custom
> runner's view of "hard" theorems.

This is the operational consequence of the RC3 rejection (`REJECT_NO_LITERAL_DELTA`) and the SX4
attribution rule: the SX3 sequence looked useful under a depth-2 custom runner, but literal RC2
already solved those cases via the same intermediate-state continuation. To avoid repeating that
over-credit, SF4 inverts the funnel — it only ever considers theorems that **literal RC2 demonstrably
fails**, then runs candidates and passes every apparent win through SX4 attribution.

## Pipeline

```
literal RC2 failures            (Part 2: collect; Part 3: live re-confirm)
  → failure clusters            (Part 4: namespace × goal-shape × symptom × name-features)
  → candidate tactics/sequences (Part 5: conservative, cluster-driven, gated, generic)
  → live evaluation             (Part 6: gated probes + depth-1 controls on confirmed failures)
  → SX4 attribution             (Part 7: credit only TRUE_DELTA over literal RC2)
  → true delta only             (Parts 8-9: family analysis + missing-lemma triage)
```

Every stage **defaults to no credit**. Only `TRUE_DELTA` (a candidate that solves a *confirmed* RC2
failure and survives SX4 attribution) advances. Nothing is promoted in SF4.

## Definitions

| term | meaning |
|---|---|
| `RC2_FAILURE` | theorem **not solved** by a literal RC2 production run (`finished == false`), `top_k=8`, `max_steps=8`, `hybrid_evolved`, `ns24_router` |
| `CANDIDATE_WIN` | a candidate solves a theorem that literal RC2 failed (pre-attribution; not yet credited) |
| `PRODUCTION_SUBSUMED` | literal RC2 already solves the theorem (the RC3 bug class). On a failure-first pool this should be **rare** — if it appears, suspect an open-flake or a stale baseline, not a real subsumption |
| `BASELINE_DUPLICATE` | a simple control (`simp` / `simp_all` / `aesop` / `classical <;> aesop`) solves it — not a sequence/lemma delta |
| `DEPTH1_DUPLICATE` | for a sequence `A <;> B`: `A` alone or `B` alone solves |
| `TRUE_DELTA` | candidate solves a confirmed RC2 failure **and** survives SX4 attribution (baseline fails, controls fail, no equivalent production continuation, generic, deterministic) |
| `MISSING_LEMMA_CANDIDATE` | a cluster of repeated RC2 failures whose goal shape suggests a reusable lemma not captured by any known tactic/sequence — flagged for triage, **not invented** |

## Why this avoids the RC3 over-credit

1. **Failure-first input.** The pool is literal RC2 failures, re-confirmed live. A theorem RC2 already
   solves can never enter as a "win."
2. **SX4 gate on every apparent win.** Even a candidate that solves a confirmed failure is only
   credited `TRUE_DELTA` after SX4 checks baseline-fails + depth-1-controls-fail + no-equivalent-
   production-`A`→`B`-continuation. (See `project/evolve/experiments/sx4/sx4_methodology.md`.)
3. **Conservative, generic probes.** Cluster-generated batteries only; no source-proof `rw` bridges
   unless explicitly marked `SOURCE_SPECIFIC_DIAGNOSTIC` (never credited).
4. **Default-to-no-credit** for `TRACE_INSUFFICIENT` / `NEEDS_REVIEW` / `BASELINE_DUPLICATE`.

SF4 produces *candidates and evidence*, never a promotion. Promotion still requires a separate
literal-wrapper validation (the RC3 process) gated by the SX4 harness.
