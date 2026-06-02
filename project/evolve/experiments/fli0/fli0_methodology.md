# FLI0 Methodology

## Goal restated

Turn the negative results of the RC5 hybrid search into a **structured corpus of failure
signals** that a later lemma-invention stage (FLI1) can act on. A failed proof attempt is treated
as evidence about *what is missing* (an intermediate lemma / bridge), not merely as a 0.

## Source of truth

| stage | role | completeness |
|---|---|---|
| RC5V2 | fully-attributed clean-failure set (149 eligible, 8 solved, committed attribution) | complete |
| RC5V3 | larger disjoint fresh frontier of live failures (318 eligible, 4 solved) | `PARTIAL_ARTIFACTS_AVAILABLE` (raw results only; no attribution/report) |

RC5V2 ∩ RC5V3 = ∅, so the two failure sets union without dedup conflicts (we still key by
`full_name` and record `source_stage`, marking any cross-stage duplicate should one appear).

## What we have vs. what we lack

**Have** (per theorem): statement text, file path, namespace, a 12-way boolean feature vector
(`has_iff/eq/subset/mem/disjoint/card/map_filter/tofinset/nat_arith/order/singleton/union_inter`),
freshness status, RC2 status, RC4 static status, the dynamic attempt record
(`programs_attempted`, `success`, `winning_program`, `controls[]` with errors, `failures[]` with
`{rank, tactic, outcome}`), retrieved lemmas (`top_lemmas[]` with statement text + score) and the
goal's definitional constants (`goal_defs[]`).

**Lack:** post-tactic **residual goal states** (the artifacts log tactic *outcomes*, not the goal
after each step). So FLI0 cannot quote "the goal got stuck at X". We set
`residual_goal_status = MISSING` everywhere and infer the gap from statement + features +
retrieved lemmas + which tactic families failed. This is the central limitation FLI0 hands to
FLI1, which may re-run live to capture residual goals for the chosen seeds.

## Failure inclusion / exclusion

A theorem enters the **raw failure corpus** iff: it was dynamic-eligible (CONFIRMED_RC2_FAILURE,
so RC2 failed by construction), RC4 static did not solve it, and the dynamic stage did not solve
it. We then tag, but separate, the non-math cases:

- `dynamic_result = timeout|killed` → excluded from CLEAN.
- `dynamic_result = infra_error` (RC5V3 B5 network/setup error with no live attempt) → excluded
  from CLEAN, tagged `INFRA`.
- failures whose only outcomes are `unknown_name` → tagged `UNKNOWN_NAME_OR_IMPORT`, excluded
  from CLEAN (the gap is availability, not a missing math lemma).
- `proof_failed` with a readable trace → **CLEAN_FAILURE**.

Dynamic successes and RC2/RC4-solved theorems are excluded entirely (they are not failures).

## Dynamic-result merge

- **RC5V2:** single B5 run → use its record + committed attribution.
- **RC5V3:** merge B1/B3/B5 — solved if any budget solved; else if any budget ran a real live
  attempt (`live=True, programs_attempted>0`) → `failed` (clean candidate); else `infra_error`.
  Controls (the 4 bare tactics) are captured at B1; we read them from whichever budget has them.

## Pattern taxonomy (Part 5)

Conservative, multi-label, rule-based over features + statement tokens + retrieved-lemma names +
failure outcomes. Labels: `MEMBERSHIP_BRIDGE`, `SINGLETON_CHARACTERIZATION`, `DISJOINT_BRIDGE`,
`SUBSET_BRIDGE`, `MAP_FILTER_BIND_BRIDGE`, `IFF_SPLIT`, `EXTENSIONALITY_NEEDED`,
`INDUCTION_GENERALIZATION`, `SIMP_LOOP_OR_RECURSION`, `UNKNOWN_NAME_OR_IMPORT`,
`ORDER_STRUCTURE_GAP`, `NAT_ARITH_GAP`, `LOW_SIGNAL`, `NEEDS_REVIEW`. Each case also gets a
confidence, a natural-language explanation, a **candidate lemma shape (NL)**, and a recommended
next probe family. We never assert "requires"; only "suggests / appears to need".

## Seed selection (Part 6)

Rank clean failures by: cleanliness → freshness → readable statement → readable trace →
namespace ∈ {List, Multiset, Finset, Set, light-Nat} → high-signal pattern label →
"retrieved lemmas exist but did not close" → "a similar theorem was solved". Take 20–40.

## Determinism & safety

All scripts are pure functions of on-disk artifacts (stable sort keys, no RNG, no clock, no live
Lean). Reading them repeatedly yields identical output. No protected file is read-for-write; no
production wrapper/router is touched.
